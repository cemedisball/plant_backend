# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pathlib import Path
from typing import List, Optional
import torch, time, sys, io, asyncio, threading
from PIL import Image, UnidentifiedImageError

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
# บีบอัด response
app.add_middleware(GZipMiddleware, minimum_size=500)

# ===== Model config (defer import/creation) =====
MODEL_PATH = Path(__file__).parent / "models" / "best69.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = (DEVICE == "cuda")
torch.set_num_threads(1)  # ช่วยบน CPU
SUGGESTED_IMGSZ = 416

_model = None
_names = {}
_stride = 32
_model_lock = threading.Lock()

def _valid_imgsz(imgsz: int, stride: int) -> int:
    r = (imgsz + stride - 1) // stride * stride
    return max(stride, r)

def get_model():
    """Lazy-load ULTRALYTICS YOLO model on first use."""
    global _model, _names, _stride
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"[model] loading: {MODEL_PATH}", flush=True)
                # ⬇️ import ภายในฟังก์ชัน เพื่อให้ import app เร็วที่สุด
                from ultralytics import YOLO
                m = YOLO(str(MODEL_PATH))
                m.to(DEVICE)
                # meta
                names = getattr(m, "names", None) or getattr(getattr(m, "model", None), "names", None) or {}
                stride = int(getattr(getattr(m, "model", None), "stride", [32])[0]) if hasattr(m, "model") else 32
                _names = names
                _stride = stride
                _model = m
                print(f"[model] loaded (device={DEVICE}, half={USE_HALF}, stride={_stride})", flush=True)
    return _model, _names, _stride

def _background_warmup():
    """ทำ warmup เบา ๆ แบบไม่บล็อก startup"""
    try:
        m, _, _ = get_model()
        dummy = Image.new("RGB", (SUGGESTED_IMGSZ, SUGGESTED_IMGSZ), (0, 0, 0))
        t0 = time.time()
        with torch.inference_mode():
            _ = m.predict(
                source=dummy, conf=0.25, iou=0.5, imgsz=SUGGESTED_IMGSZ,
                device=DEVICE, half=USE_HALF, verbose=False
            )
        print(f"[warmup] {time.time()-t0:.2f}s (device={DEVICE}, half={USE_HALF})", flush=True)
    except Exception as e:
        print(f"[warmup] skipped: {e}", file=sys.stderr, flush=True)

@app.on_event("startup")
def on_startup():
    # ❌ ไม่โหลดโมเดล/ไม่ warmup แบบบล็อก
    # ✅ ทำแบบ background เพื่อให้ health check ผ่านทันที
    threading.Thread(target=_background_warmup, daemon=True).start()
    print("[startup] ready", flush=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"message": "Backend is running!", "device": DEVICE, "half": USE_HALF}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.5, ge=0.0, le=1.0),
    imgsz: int = Query(640, description="inference image size"),
    max_det: int = Query(100, description="max detections per image"),
    classes: Optional[List[int]] = Query(None),
    tta: bool = Query(False, description="test-time augmentation"),
    agnostic_nms: bool = Query(True),
    top_k: int = Query(5, ge=1, le=100, description="return top-K results"),
    return_boxes: bool = Query(False, description="include bbox fields"),
    infer_timeout_s: int = Query(30, ge=5, le=120, description="server-side inference timeout (sec)"),
):
    ctype = (file.content_type or "").lower()
    if not ctype.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    # ⬇️ ดึงโมเดล (โหลดจริงครั้งแรกที่มีคำขอ)
    model, names_meta, stride = get_model()
    imgsz_eff = _valid_imgsz(imgsz, stride)

    # ย้ายงานบล็อก I/O/CPU ออกไปทำใน threadpool
    loop = asyncio.get_running_loop()
    def _blocking_infer():
        with torch.inference_mode():
            return model.predict(
                source=image, conf=conf, iou=iou, imgsz=imgsz_eff,
                max_det=max_det, classes=classes, device=DEVICE,
                augment=tta, agnostic_nms=agnostic_nms, half=USE_HALF,
                verbose=False,
            )

    try:
        results = await asyncio.wait_for(
            loop.run_in_executor(None, _blocking_infer),
            timeout=infer_timeout_s
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Inference timeout ({infer_timeout_s}s)")

    preds = []
    for r in results:
        names = r.names or names_meta
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            conf_val = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            item = {
                "confidence": conf_val,
                "class": cls_id,
                "name": (names.get(cls_id) if isinstance(names, dict) else str(cls_id)),
            }
            if return_boxes:
                item.update({
                    "xmin": float(xyxy[0]), "ymin": float(xyxy[1]),
                    "xmax": float(xyxy[2]), "ymax": float(xyxy[3]),
                })
            preds.append(item)

    preds.sort(key=lambda x: x["confidence"], reverse=True)
    preds = preds[:top_k]

    return {
        "predictions": preds,
        "params": {
            "conf": conf, "iou": iou, "imgsz": imgsz_eff, "max_det": max_det,
            "classes": classes, "tta": tta, "agnostic_nms": agnostic_nms,
            "device": DEVICE, "half": USE_HALF, "stride": stride,
            "top_k": top_k, "return_boxes": return_boxes,
        },
    }

if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[startup] Starting server on 0.0.0.0:{port}", flush=True)
    uvicorn.run("app:app", host="0.0.0.0", port=port)
