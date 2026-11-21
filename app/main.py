"""AI Sentinel (Iris Vision) - FastAPI REST API with 2-Layer Defense-in-Depth OOD Rejection.

Startup loads:
1. Layer 1 Binary Gatekeeper (`models/gatekeeper_best.pth`, MobileNetV2: `non_eye` vs `valid_eye`).
2. Layer 2 Main 5-Class Classifier (`models/ai_sentinel_5class_best.pth`, fallback `models/ai_sentinel_best.pth`).

Endpoints:
- GET /health: Server status, loaded models, defense-in-depth status, and GPU/CPU compute state.
- POST /api/v1/predict:
  1. Technical validation (`FundusImage` magic bytes, JPEG/PNG, >=224x224, EXIF strip) & Laplacian blur check.
  2. Single preprocessing (`safe_crop_fundus_roi` + `224x224` + LAB CLAHE).
  3. Layer 1 Gatekeeper check (`P(valid_eye) < 0.65` -> HTTP 422 `rejected_ood`).
  4. Layer 2 5-Class check (`argmax == non_fundus` or `P(non_fundus) > 0.40` -> HTTP 422 `rejected_ood`).
  5. Medical 4-class probability renormalization, 90% confidence thresholding, Grad-CAM Base64,
     and `gatekeeper_eye_score` audit field.
- GET /: Interactive diagnostic web UI with OOD rejection banner & Gatekeeper score badge.
"""

import base64
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import time
from typing import Dict, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ALLOWED_API_FORMATS,
    BLUR_LAPLACIAN_THRESHOLD,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    GATEKEEPER_BEST_PATH,
    GATEKEEPER_CONFIG_JSON,
    GATEKEEPER_EYE_THRESHOLD,
    IMAGE_SIZE,
    L2_NON_FUNDUS_THRESHOLD,
    MIN_IMAGE_DIMENSION,
    MODEL_5CLASS_PATH,
    MODEL_BEST_PATH,
    MODEL_CONFIG_JSON,
    OOD_CLASS_NAME,
    USE_AMP,
    get_device,
)
from src.dataset import get_default_transforms
from src.evaluate import GradCAM, load_trained_model
from src.image_validator import FundusImage, ImageValidationError
from src.preprocess import preprocess_image

app_state: dict = {
    "model": None,
    "model_path": None,
    "model_classes": CLASSES,
    "config": {},
    "gatekeeper_model": None,
    "gatekeeper_config": {},
    "device": None,
    "cam_extractor": None,
    "transform": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Layer 1 Gatekeeper and Layer 2 5-Class diagnostic model on startup."""
    device = get_device()
    app_state["device"] = device
    app_state["transform"] = get_default_transforms(is_train=False)

    # 1. Load Layer 1 Binary Gatekeeper if available
    if GATEKEEPER_BEST_PATH.exists():
        gk_model, gk_cfg = load_trained_model(
            GATEKEEPER_BEST_PATH, GATEKEEPER_CONFIG_JSON, device=device
        )
        app_state["gatekeeper_model"] = gk_model
        app_state["gatekeeper_config"] = gk_cfg
        dummy = torch.zeros((1, 3, *IMAGE_SIZE), device=device)
        with torch.no_grad():
            gk_model(dummy)
        print(
            f"[INFO] Layer 1 Gatekeeper Loaded ({gk_cfg.get('backbone', 'mobilenet_v2')}) "
            f"| Threshold P(valid_eye) >= {GATEKEEPER_EYE_THRESHOLD}"
        )
    else:
        print(f"[WARN] Gatekeeper checkpoint not found at {GATEKEEPER_BEST_PATH}.")

    # 2. Load Layer 2 Main Diagnostic Model (prefer 5-class checkpoint, fallback to 4-class)
    active_ckpt = MODEL_5CLASS_PATH if MODEL_5CLASS_PATH.exists() else MODEL_BEST_PATH
    if not active_ckpt.exists():
        print(f"[WARN] Main checkpoint not found at {active_ckpt}. Starting in degraded mode.")
    else:
        model, config = load_trained_model(active_ckpt, MODEL_CONFIG_JSON, device=device)
        app_state["model"] = model
        app_state["model_path"] = active_ckpt
        app_state["config"] = config
        app_state["model_classes"] = list(config.get("classes", CLASSES))
        target_layer = model.get_target_layer_for_cam()
        app_state["cam_extractor"] = GradCAM(model, target_layer)

        dummy = torch.zeros((1, 3, *IMAGE_SIZE), device=device)
        with torch.no_grad():
            model(dummy)
        print(
            f"[INFO] Layer 2 Main Model Loaded: {active_ckpt.name} "
            f"({config.get('backbone', 'resnet50')}, classes={app_state['model_classes']}) "
            f"on {device} | Confidence Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%"
        )

    yield

    if app_state["cam_extractor"] is not None:
        app_state["cam_extractor"].close()
    print("[INFO] AI Sentinel API Shutdown Cleanly.")


app = FastAPI(
    title="AI Sentinel - Iris Vision API",
    description=(
        "REST API for Ocular Disease Classification with 2-Layer Defense-in-Depth "
        "Out-of-Distribution (OOD) Rejection & Grad-CAM Explainability"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic Schemas
# ==============================================================================
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_name: str = Field(..., examples=["ai_sentinel_5class_best.pth"])
    backbone: str = Field(..., examples=["resnet50"])
    device: str = Field(..., examples=["cuda"])
    gpu_name: Optional[str] = Field(None, examples=["NVIDIA GeForce RTX 3050 6GB Laptop GPU"])
    gpu_memory_allocated_mb: Optional[float] = Field(None, examples=[42.5])
    model_loaded: bool = Field(..., examples=[True])
    gatekeeper_loaded: bool = Field(default=False, examples=[True])
    ood_defense_enabled: bool = Field(default=False, examples=[True])
    gatekeeper_threshold: float = Field(default=GATEKEEPER_EYE_THRESHOLD, examples=[0.65])
    l2_non_fundus_threshold: float = Field(default=L2_NON_FUNDUS_THRESHOLD, examples=[0.40])
    confidence_threshold: float = Field(..., examples=[0.90])
    classes: list[str] = Field(..., examples=[CLASSES])
    model_classes: list[str] = Field(default_factory=lambda: CLASSES)
    version: str = Field(..., examples=["2.0.0"])


class BlurAssessment(BaseModel):
    laplacian_variance: float = Field(..., examples=[215.4])
    is_blurry: bool = Field(..., examples=[False])
    threshold: float = Field(..., examples=[100.0])
    warning: Optional[str] = Field(None)


class PredictionResponse(BaseModel):
    status: str = Field(default="success")
    filename: str = Field(..., examples=["104.jpg"])
    predicted_class: str = Field(..., examples=["cataract"])
    confidence: float = Field(..., examples=[0.9842])
    confidence_percentage: str = Field(..., examples=["98.42%"])
    confidence_threshold: float = Field(..., examples=[0.90])
    is_confident: bool = Field(..., examples=[True])
    diagnosis_status: str = Field(..., examples=["confident"])
    gatekeeper_eye_score: Optional[float] = Field(
        None,
        description="Layer 1 Gatekeeper probability that the uploaded image is a valid ocular image",
        examples=[0.9964],
    )
    non_fundus_score: Optional[float] = Field(
        None,
        description="Layer 2 5-class raw probability for non_fundus class",
        examples=[0.0012],
    )
    blur_assessment: BlurAssessment
    probabilities: Dict[str, float]
    gradcam_base64: str = Field(..., description="Base64-encoded JPEG Grad-CAM overlay")
    inference_time_ms: float = Field(..., examples=[18.4])
    clinical_notes: Optional[str] = Field(None)


# ==============================================================================
# Validation, Blur Detection, & Grad-CAM Helpers
# ==============================================================================
def validate_and_decode_image(file: UploadFile, file_bytes: bytes) -> np.ndarray:
    """Validate uploaded image via FundusImage (magic bytes, integrity, >=224x224, EXIF strip)."""
    fundus_img = FundusImage(file_bytes, filename=file.filename)
    try:
        fundus_img.validateFormat(
            allowed_formats=ALLOWED_API_FORMATS,
            min_width=MIN_IMAGE_DIMENSION,
            min_height=MIN_IMAGE_DIMENSION,
            check_extension=True,
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc

    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. Expected an image file.",
        )

    if fundus_img.has_exif():
        fundus_img.stripExif()

    return fundus_img.to_bgr_array()


def assess_image_blur(img_bgr: np.ndarray) -> BlurAssessment:
    """Compute Laplacian variance on grayscale image to flag potential motion/focus blur."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    is_blurry = bool(lap_var < BLUR_LAPLACIAN_THRESHOLD)
    warning = (
        f"Image sharpness is low (Laplacian variance {lap_var:.1f} < {BLUR_LAPLACIAN_THRESHOLD:.1f}). "
        "Consider capturing a sharper photo for higher diagnostic reliability."
        if is_blurry
        else None
    )
    return BlurAssessment(
        laplacian_variance=lap_var,
        is_blurry=is_blurry,
        threshold=BLUR_LAPLACIAN_THRESHOLD,
        warning=warning,
    )


def generate_gradcam_base64(
    prep_rgb: np.ndarray,
    target_class_idx: int,
) -> str:
    """Generate Grad-CAM overlay and encode as Base64 JPEG data URL."""
    cam_extractor = app_state["cam_extractor"]
    device = app_state["device"]
    transform = app_state["transform"]

    input_tensor = transform(Image.fromarray(prep_rgb)).unsqueeze(0).to(device)
    cam_map, _, _ = cam_extractor.generate(input_tensor, class_idx=target_class_idx)

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = np.clip(
        0.58 * prep_rgb.astype(np.float32) + 0.42 * heatmap_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    success, encoded_jpg = cv2.imencode(
        ".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode Grad-CAM heatmap image.",
        )

    return "data:image/jpeg;base64," + base64.b64encode(encoded_jpg.tobytes()).decode("utf-8")


# ==============================================================================
# API Endpoints
# ==============================================================================
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return server status, loaded model details, defense-in-depth status, and GPU/CPU state."""
    device = app_state["device"]
    is_cuda = (device is not None) and (device.type == "cuda")
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else None
    mem_mb = (
        round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if is_cuda else None
    )
    model_path = app_state.get("model_path") or MODEL_BEST_PATH
    model_classes = app_state.get("model_classes", CLASSES)
    gk_loaded = app_state.get("gatekeeper_model") is not None
    ood_enabled = gk_loaded or (OOD_CLASS_NAME in model_classes)

    return HealthResponse(
        status="healthy" if app_state["model"] is not None else "degraded",
        model_name=model_path.name,
        backbone=app_state["config"].get("backbone", "resnet50"),
        device=str(device) if device else "unknown",
        gpu_name=gpu_name,
        gpu_memory_allocated_mb=mem_mb,
        model_loaded=app_state["model"] is not None,
        gatekeeper_loaded=gk_loaded,
        ood_defense_enabled=ood_enabled,
        gatekeeper_threshold=GATEKEEPER_EYE_THRESHOLD,
        l2_non_fundus_threshold=L2_NON_FUNDUS_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        classes=CLASSES,
        model_classes=model_classes,
        version="2.0.0",
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_ocular_disease(
    file: UploadFile = File(..., description="Ocular image file (JPG, PNG)"),
):
    """Run 2-Layer Defense-in-Depth OOD check followed by 4-class ocular disease diagnosis."""
    model = app_state["model"]
    gk_model = app_state.get("gatekeeper_model")
    model_classes: list[str] = app_state.get("model_classes", CLASSES)
    device = app_state["device"]
    transform = app_state["transform"]

    if model is None or app_state["cam_extractor"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoint is not loaded on server.",
        )

    t0 = time.time()
    file_bytes = await file.read()
    img_bgr = validate_and_decode_image(file, file_bytes)

    # 1. Laplacian Variance Blur Detection
    blur_assessment = assess_image_blur(img_bgr)

    # 2. Single Preprocessing Pass: safe_crop_fundus_roi + Resize 224x224 + LAB CLAHE
    prep_rgb = preprocess_image(
        img_bgr, class_name=None, target_size=IMAGE_SIZE, return_rgb=True
    )
    input_tensor = transform(Image.fromarray(prep_rgb)).unsqueeze(0).to(device)
    use_amp = (device.type == "cuda") and USE_AMP

    # 3. LAYER 1: Binary Gatekeeper Check (Opsi 2)
    gatekeeper_eye_score: Optional[float] = None
    if gk_model is not None:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                gk_logits = gk_model(input_tensor)
                gk_probs = F.softmax(gk_logits, dim=1).squeeze(0)
        gatekeeper_eye_score = round(float(gk_probs[1].item()), 4)

        if gatekeeper_eye_score < GATEKEEPER_EYE_THRESHOLD:
            elapsed_ms = round((time.time() - t0) * 1000.0, 2)
            msg = "Gambar ditolak oleh Gatekeeper: Citra yang diunggah bukan foto pemeriksaan mata/fundus."
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "status": "rejected_ood",
                    "rejected_by": "layer1_gatekeeper",
                    "filename": file.filename or "uploaded_image",
                    "detail": msg,
                    "message": msg,
                    "gatekeeper_eye_score": gatekeeper_eye_score,
                    "gatekeeper_threshold": GATEKEEPER_EYE_THRESHOLD,
                    "inference_time_ms": elapsed_ms,
                },
            )

    # 4. LAYER 2: Main 5-Class Model Inference (Opsi 1)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_tensor)
            raw_probs = F.softmax(logits, dim=1).squeeze(0)

    raw_pred_idx = int(torch.argmax(raw_probs).item())
    raw_pred_class = model_classes[raw_pred_idx]
    non_fundus_score: Optional[float] = None

    if OOD_CLASS_NAME in model_classes:
        ood_idx = model_classes.index(OOD_CLASS_NAME)
        non_fundus_score = round(float(raw_probs[ood_idx].item()), 4)
        if gatekeeper_eye_score is None:
            gatekeeper_eye_score = round(1.0 - non_fundus_score, 4)

        if raw_pred_class == OOD_CLASS_NAME or non_fundus_score > L2_NON_FUNDUS_THRESHOLD:
            elapsed_ms = round((time.time() - t0) * 1000.0, 2)
            msg = (
                "Gambar ditolak oleh Model Utama: Citra tidak memenuhi kriteria "
                "foto fundus retina atau pemeriksaan lensa mata."
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "status": "rejected_ood",
                    "rejected_by": "layer2_5class_non_fundus",
                    "filename": file.filename or "uploaded_image",
                    "detail": msg,
                    "message": msg,
                    "gatekeeper_eye_score": gatekeeper_eye_score,
                    "non_fundus_score": non_fundus_score,
                    "non_fundus_threshold": L2_NON_FUNDUS_THRESHOLD,
                    "inference_time_ms": elapsed_ms,
                },
            )

    # 5. Renormalize Probabilities Across the 4 Medical Classes
    med_raw = {
        cls_name: float(raw_probs[model_classes.index(cls_name)].item())
        for cls_name in CLASSES
        if cls_name in model_classes
    }
    med_sum = max(1e-8, sum(med_raw.values()))
    prob_dict = {
        cls_name: round(med_raw[cls_name] / med_sum, 4)
        for cls_name in CLASSES
    }
    # Ensure exact 1.0 sum after 4-decimal rounding
    predicted_class = max(med_raw, key=med_raw.get)
    pred_conf = float(med_raw[predicted_class] / med_sum)

    # 6. Enforce Confidence Threshold (90%)
    is_confident = bool(pred_conf >= CONFIDENCE_THRESHOLD)
    diagnosis_status = "confident" if is_confident else "uncertain"

    notes_parts = []
    if not is_confident:
        notes_parts.append(
            f"Prediction confidence ({pred_conf * 100:.1f}%) is below the {CONFIDENCE_THRESHOLD * 100:.0f}% threshold. "
            "Re-imaging or specialized ophthalmologic review is recommended."
        )
    if blur_assessment.warning:
        notes_parts.append(blur_assessment.warning)
    clinical_notes = " ".join(notes_parts) if notes_parts else None

    # 7. Grad-CAM Heatmap Overlay using raw index in loaded model (5-class safe)
    target_raw_idx = model_classes.index(predicted_class)
    gradcam_b64 = generate_gradcam_base64(prep_rgb, target_raw_idx)

    elapsed_ms = round((time.time() - t0) * 1000.0, 2)

    return PredictionResponse(
        status="success",
        filename=file.filename or "uploaded_image",
        predicted_class=predicted_class,
        confidence=round(pred_conf, 4),
        confidence_percentage=f"{pred_conf * 100:.2f}%",
        confidence_threshold=CONFIDENCE_THRESHOLD,
        is_confident=is_confident,
        diagnosis_status=diagnosis_status,
        gatekeeper_eye_score=gatekeeper_eye_score,
        non_fundus_score=non_fundus_score,
        blur_assessment=blur_assessment,
        probabilities=prob_dict,
        gradcam_base64=gradcam_b64,
        inference_time_ms=elapsed_ms,
        clinical_notes=clinical_notes,
    )


# ==============================================================================
# Interactive Web Dashboard
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> str:
    """Serve interactive diagnostic UI calling POST /api/v1/predict."""
    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU (Windows 11 Native)"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Sentinel — Iris Vision Ocular Diagnostics</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .dragover {{ border-color: #38bdf8 !important; background-color: rgba(14, 165, 233, 0.08) !important; }}
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
  <div class="max-w-6xl mx-auto px-4 py-8">
    <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-slate-800 gap-4">
      <div class="flex items-center gap-3">
        <div class="h-11 w-11 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-2xl">👁️</div>
        <div>
          <h1 class="text-2xl font-black tracking-tight text-white">AI Sentinel · Iris Vision</h1>
          <p class="text-xs text-slate-400">2-Layer Defense-in-Depth OOD Guard &amp; Ocular Disease Classification (Threshold: {int(CONFIDENCE_THRESHOLD * 100)}%)</p>
        </div>
      </div>
      <div class="flex flex-wrap items-center gap-2">
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span> {device_name}
        </span>
        <a href="/docs" class="px-3 py-1 rounded-full text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">Swagger /docs</a>
      </div>
    </header>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 mt-8">
      <div class="lg:col-span-5 space-y-6">
        <div class="bg-slate-900/70 rounded-2xl p-6 border border-slate-800 shadow-xl">
          <h2 class="text-base font-bold text-white mb-4">📥 Upload Ocular Image</h2>
          <div id="drop-zone" class="border-2 border-dashed border-slate-700 rounded-xl p-6 text-center cursor-pointer hover:border-slate-500 transition-all bg-slate-950/50">
            <input type="file" id="file-input" accept="image/jpeg,image/png" class="hidden">
            <div id="drop-placeholder" class="space-y-2 py-4">
              <div class="text-3xl">📸</div>
              <p class="text-sm font-medium text-slate-200">Drag &amp; drop fundus or eye photo</p>
              <p class="text-xs text-slate-400">or click to browse (JPEG, PNG &ge; 224&times;224 px)</p>
            </div>
            <img id="image-preview" class="hidden max-h-64 mx-auto rounded-lg border border-slate-700 object-cover" alt="Preview">
          </div>

          <div class="mt-4 flex gap-3">
            <button id="predict-btn" disabled class="flex-1 py-3 px-4 rounded-xl font-semibold text-white bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg shadow-cyan-600/25 flex items-center justify-center gap-2">
              <span id="btn-spinner" class="hidden animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span>
              <span id="btn-text">Analyze Image</span>
            </button>
            <button id="clear-btn" class="py-3 px-4 rounded-xl font-semibold text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700">Clear</button>
          </div>
        </div>
      </div>

      <div class="lg:col-span-7">
        <div class="bg-slate-900/70 rounded-2xl p-6 border border-slate-800 shadow-xl min-h-[460px] flex flex-col justify-center">
          <div id="empty-state" class="text-center py-16 text-slate-500">
            <div class="text-5xl mb-3">🩺</div>
            <p class="text-base font-medium">Awaiting Input Image</p>
            <p class="text-xs mt-1">Protected by Layer-1 Binary Gatekeeper &amp; Layer-2 5-Class OOD Filter</p>
          </div>

          <div id="ood-alert" class="hidden p-5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 space-y-2">
            <div class="text-base font-bold text-rose-400">⛔ Out-of-Distribution Image Rejected</div>
            <p id="ood-detail" class="text-xs"></p>
            <p id="ood-score" class="text-xs font-mono text-rose-300/80"></p>
          </div>

          <div id="result-content" class="hidden space-y-5">
            <div class="flex flex-wrap items-center justify-between gap-3 p-4 rounded-xl bg-slate-950/70 border border-slate-800">
              <div>
                <p class="text-[11px] uppercase tracking-wider text-slate-400 font-bold">Predicted Diagnosis</p>
                <div class="flex items-center gap-2.5 mt-1">
                  <h3 id="res-class" class="text-2xl font-black capitalize text-white">---</h3>
                  <span id="res-status-badge" class="px-2.5 py-0.5 rounded-full text-xs font-bold"></span>
                </div>
              </div>
              <div class="text-right">
                <p class="text-xs text-slate-400">Confidence</p>
                <div id="res-conf" class="text-2xl font-black text-cyan-400">--%</div>
              </div>
            </div>

            <div id="res-notes" class="hidden p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs"></div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              <div class="flex items-center justify-between px-3 py-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-slate-400">Sharpness (Laplacian):</span>
                <span id="res-blur" class="font-mono font-semibold">---</span>
              </div>
              <div class="flex items-center justify-between px-3 py-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span class="text-slate-400">Gatekeeper Eye Score:</span>
                <span id="res-gk" class="font-mono font-semibold text-emerald-400">---</span>
              </div>
            </div>

            <div>
              <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Class Probability Distribution</h4>
              <div id="prob-bars" class="space-y-3"></div>
            </div>

            <div class="pt-4 border-t border-slate-800">
              <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Grad-CAM Visual Attention Map</h4>
              <div class="bg-slate-950 rounded-xl p-3 border border-slate-800 flex justify-center">
                <img id="res-gradcam" class="max-h-72 rounded-lg" alt="Grad-CAM Overlay">
              </div>
            </div>

            <div class="text-[11px] text-slate-500 text-right font-mono" id="res-timing">Latency: -- ms</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const dropPlaceholder = document.getElementById('drop-placeholder');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');
    const emptyState = document.getElementById('empty-state');
    const oodAlert = document.getElementById('ood-alert');
    const resultContent = document.getElementById('result-content');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {{ e.preventDefault(); dropZone.classList.add('dragover'); }});
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {{
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    }});
    fileInput.addEventListener('change', (e) => {{ if (e.target.files.length) handleFile(e.target.files[0]); }});

    function handleFile(file) {{
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {{
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        dropPlaceholder.classList.add('hidden');
        predictBtn.disabled = false;
      }};
      reader.readAsDataURL(file);
    }}

    clearBtn.addEventListener('click', () => {{
      selectedFile = null;
      fileInput.value = '';
      imagePreview.classList.add('hidden');
      dropPlaceholder.classList.remove('hidden');
      predictBtn.disabled = true;
      emptyState.classList.remove('hidden');
      oodAlert.classList.add('hidden');
      resultContent.classList.add('hidden');
    }});

    predictBtn.addEventListener('click', async () => {{
      if (!selectedFile) return;
      predictBtn.disabled = true;
      btnSpinner.classList.remove('hidden');
      btnText.textContent = 'Analyzing...';

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {{
        const response = await fetch('/api/v1/predict', {{ method: 'POST', body: formData }});
        const data = await response.json();
        if (response.status === 422 && data.status === 'rejected_ood') {{
          emptyState.classList.add('hidden');
          resultContent.classList.add('hidden');
          oodAlert.classList.remove('hidden');
          document.getElementById('ood-detail').textContent = data.detail;
          document.getElementById('ood-score').textContent =
            `Rejected by: ${{data.rejected_by}} | Gatekeeper Eye Score: ${{(data.gatekeeper_eye_score * 100).toFixed(2)}}%`;
          return;
        }}
        if (!response.ok) throw new Error(data.detail || response.statusText);
        renderResults(data);
      }} catch (err) {{
        alert('Error: ' + err.message);
      }} finally {{
        predictBtn.disabled = false;
        btnSpinner.classList.add('hidden');
        btnText.textContent = 'Analyze Image';
      }}
    }});

    function renderResults(data) {{
      emptyState.classList.add('hidden');
      oodAlert.classList.add('hidden');
      resultContent.classList.remove('hidden');

      document.getElementById('res-class').textContent = data.predicted_class.replace('_', ' ');
      document.getElementById('res-conf').textContent = data.confidence_percentage;

      const badge = document.getElementById('res-status-badge');
      if (data.is_confident) {{
        badge.textContent = 'CONFIDENT (>={int(CONFIDENCE_THRESHOLD * 100)}%)';
        badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
      }} else {{
        badge.textContent = 'UNCERTAIN (<{int(CONFIDENCE_THRESHOLD * 100)}%)';
        badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30';
      }}

      const notesEl = document.getElementById('res-notes');
      if (data.clinical_notes) {{
        notesEl.textContent = '⚠️ ' + data.clinical_notes;
        notesEl.classList.remove('hidden');
      }} else {{
        notesEl.classList.add('hidden');
      }}

      const blurEl = document.getElementById('res-blur');
      const b = data.blur_assessment;
      blurEl.textContent = `${{b.laplacian_variance.toFixed(1)}} (${{b.is_blurry ? 'BLURRY' : 'SHARP'}})`;
      blurEl.className = b.is_blurry ? 'font-mono font-semibold text-amber-400' : 'font-mono font-semibold text-emerald-400';

      const gkEl = document.getElementById('res-gk');
      gkEl.textContent = data.gatekeeper_eye_score != null ? `${{(data.gatekeeper_eye_score * 100).toFixed(2)}}%` : 'N/A';

      const bars = document.getElementById('prob-bars');
      bars.innerHTML = '';
      for (const [cls, prob] of Object.entries(data.probabilities)) {{
        const pct = (prob * 100).toFixed(1);
        const isPred = cls === data.predicted_class;
        bars.innerHTML += `
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="capitalize ${{isPred ? 'text-white font-bold' : 'text-slate-400'}}">${{cls.replace('_', ' ')}}</span>
              <span class="${{isPred ? 'text-cyan-400 font-bold' : 'text-slate-400'}}">${{pct}}%</span>
            </div>
            <div class="w-full bg-slate-950 rounded-full h-2 overflow-hidden">
              <div class="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full" style="width: ${{pct}}%"></div>
            </div>
          </div>`;
      }}

      document.getElementById('res-gradcam').src = data.gradcam_base64;
      document.getElementById('res-timing').textContent = `Inference Latency: ${{data.inference_time_ms}} ms`;
    }}
  </script>
</body>
</html>
"""
