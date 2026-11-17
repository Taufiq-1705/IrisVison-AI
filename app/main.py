"""AI Sentinel (Iris Vision) - FastAPI REST API & Diagnostic Interface.

Startup loads `models/ai_sentinel_best.pth` and `models/model_config.json`.
Endpoints:
- GET /health: Server status, model name/backbone, and GPU/CPU compute status.
- POST /api/v1/predict: Image validation, Laplacian blur detection,
  safe_crop_fundus_roi + LAB CLAHE preprocessing, 4-class Softmax inference,
  80% confidence thresholding, and Base64 Grad-CAM heatmap generation.
- GET /: Interactive diagnostic web UI.
"""

import base64
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BLUR_LAPLACIAN_THRESHOLD,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_BEST_PATH,
    MODEL_CONFIG_JSON,
    USE_AMP,
    VALID_EXTENSIONS,
    get_device,
)
from src.dataset import get_default_transforms
from src.evaluate import GradCAM, load_trained_model
from src.preprocess import preprocess_image

app_state: dict = {
    "model": None,
    "config": {},
    "device": None,
    "cam_extractor": None,
    "transform": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights and config on startup; release hooks on shutdown."""
    device = get_device()
    app_state["device"] = device
    app_state["transform"] = get_default_transforms(is_train=False)

    if not MODEL_BEST_PATH.exists():
        print(f"[WARN] Checkpoint not found at {MODEL_BEST_PATH}. Starting in degraded mode.")
    else:
        model, config = load_trained_model(MODEL_BEST_PATH, MODEL_CONFIG_JSON, device=device)
        app_state["model"] = model
        app_state["config"] = config
        target_layer = model.get_target_layer_for_cam()
        app_state["cam_extractor"] = GradCAM(model, target_layer)

        # Warm-up CUDA/CPU inference pipeline
        dummy = torch.zeros((1, 3, *IMAGE_SIZE), device=device)
        with torch.no_grad():
            model(dummy)
        print(
            f"[INFO] AI Sentinel Loaded ({config.get('backbone', 'mobilenet_v2')}) "
            f"on {device} | Confidence Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%"
        )

    yield

    if app_state["cam_extractor"] is not None:
        app_state["cam_extractor"].close()
    print("[INFO] AI Sentinel API Shutdown Cleanly.")


app = FastAPI(
    title="AI Sentinel - Iris Vision API",
    description="REST API for 4-Class Ocular Disease Classification & Grad-CAM Explainability",
    version="1.0.0",
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
    model_name: str = Field(..., examples=["ai_sentinel_best.pth"])
    backbone: str = Field(..., examples=["mobilenet_v2"])
    device: str = Field(..., examples=["cuda"])
    gpu_name: Optional[str] = Field(None, examples=["NVIDIA GeForce RTX 3050 6GB Laptop GPU"])
    gpu_memory_allocated_mb: Optional[float] = Field(None, examples=[14.2])
    model_loaded: bool = Field(..., examples=[True])
    confidence_threshold: float = Field(..., examples=[0.80])
    classes: list[str] = Field(..., examples=[CLASSES])
    version: str = Field(..., examples=["1.0.0"])


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
    confidence_threshold: float = Field(..., examples=[0.80])
    is_confident: bool = Field(..., examples=[True])
    diagnosis_status: str = Field(..., examples=["confident"])
    blur_assessment: BlurAssessment
    probabilities: Dict[str, float]
    gradcam_base64: str = Field(..., description="Base64-encoded JPEG Grad-CAM overlay")
    inference_time_ms: float = Field(..., examples=[18.4])
    clinical_notes: Optional[str] = Field(None)


# ==============================================================================
# Validation, Blur Detection, & Grad-CAM Helpers
# ==============================================================================
def validate_and_decode_image(file: UploadFile, file_bytes: bytes) -> np.ndarray:
    """Validate uploaded image extension/MIME and decode into OpenCV BGR array."""
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in VALID_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(VALID_EXTENSIONS)}",
            )

    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. Expected an image file.",
        )

    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode image data. Please upload a valid JPG, PNG, BMP, or WebP image.",
        )
    return img_bgr


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
    """Return server status, loaded model details, and GPU/CPU compute state."""
    device = app_state["device"]
    is_cuda = (device is not None) and (device.type == "cuda")
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else None
    mem_mb = (
        round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if is_cuda else None
    )

    return HealthResponse(
        status="healthy" if app_state["model"] is not None else "degraded",
        model_name=MODEL_BEST_PATH.name,
        backbone=app_state["config"].get("backbone", "mobilenet_v2"),
        device=str(device) if device else "unknown",
        gpu_name=gpu_name,
        gpu_memory_allocated_mb=mem_mb,
        model_loaded=app_state["model"] is not None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        classes=CLASSES,
        version="1.0.0",
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_ocular_disease(
    file: UploadFile = File(..., description="Ocular image file (JPG, PNG, BMP, WebP)"),
) -> PredictionResponse:
    """Classify 4 ocular diseases with blur detection, 80% threshold, and Base64 Grad-CAM."""
    model = app_state["model"]
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

    # 2. Preprocessing: safe_crop_fundus_roi + Resize 224x224 + LAB CLAHE
    prep_rgb = preprocess_image(
        img_bgr, class_name=None, target_size=IMAGE_SIZE, return_rgb=True
    )

    # 3. Inference with Softmax
    input_tensor = transform(Image.fromarray(prep_rgb)).unsqueeze(0).to(device)
    use_amp = (device.type == "cuda") and USE_AMP
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())
    predicted_class = CLASSES[pred_idx]

    prob_dict = {
        cls_name: round(float(probs[i].item()), 4)
        for i, cls_name in enumerate(CLASSES)
    }

    # 4. Enforce 80% Confidence Threshold
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

    # 5. Grad-CAM Heatmap Overlay (Base64 JPEG)
    gradcam_b64 = generate_gradcam_base64(prep_rgb, pred_idx)

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
          <p class="text-xs text-slate-400">Ocular Disease Classification &amp; Grad-CAM Explainability (Threshold: {int(CONFIDENCE_THRESHOLD * 100)}%)</p>
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
            <input type="file" id="file-input" accept="image/*" class="hidden">
            <div id="drop-placeholder" class="space-y-2 py-4">
              <div class="text-3xl">📸</div>
              <p class="text-sm font-medium text-slate-200">Drag &amp; drop fundus or eye photo</p>
              <p class="text-xs text-slate-400">or click to browse (JPG, PNG, BMP, WebP)</p>
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
            <p class="text-xs mt-1">Results include Softmax breakdown, blur detection, 80% confidence gate &amp; Grad-CAM heatmap</p>
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

            <div class="flex items-center justify-between text-xs px-3 py-2 rounded-lg bg-slate-950/60 border border-slate-800">
              <span class="text-slate-400">Sharpness Check (Laplacian Variance):</span>
              <span id="res-blur" class="font-mono font-semibold">---</span>
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
      resultContent.classList.remove('hidden');

      document.getElementById('res-class').textContent = data.predicted_class.replace('_', ' ');
      document.getElementById('res-conf').textContent = data.confidence_percentage;

      const badge = document.getElementById('res-status-badge');
      if (data.is_confident) {{
        badge.textContent = 'CONFIDENT (>=80%)';
        badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
      }} else {{
        badge.textContent = 'UNCERTAIN (<80%)';
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

