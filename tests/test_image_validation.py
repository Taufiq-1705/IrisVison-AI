"""Unit and integration tests (pytest) for FundusImage validation & FastAPI security boundaries.

Covers:
1. `FundusImage` attributes & methods (`format`, `lebarPx`, `tinggiPx`, `sha256`,
   `validateFormat()`, `computeHash()`, `stripExif()`, `to_bgr_array()`).
2. Format compatibility: Lossless PNG vs Lossy JPEG (Q=95), default {"JPEG", "PNG"}
   vs explicit {"JPEG", "PNG", "WEBP", "BMP"} in robustness mode.
3. Channel robustness: 4-channel RGBA PNG & 1-channel Grayscale PNG -> 3-channel RGB tensor.
4. Security & negative boundaries on `POST /api/v1/predict`:
   - 0-byte empty file -> HTTP 400
   - Truncated JPEG stream -> HTTP 400
   - Magic-byte spoofing (PDF/text renamed to `mata_pasien.jpg`) -> HTTP 400
   - Resolution below 224x224 (`128x128`) -> HTTP 400
   - EXIF/GPS metadata de-identification via `stripExif()`.
"""

import io
from pathlib import Path
import sys

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from src.config import CLASSES, CLEAN_MANIFEST_CSV, CONFIDENCE_THRESHOLD, DATASET_DIR
from src.image_validator import FundusImage, ImageValidationError
from src.preprocess import preprocess_image


# ==============================================================================
# Fixtures & Helpers
# ==============================================================================
def _get_sample_test_image_path() -> Path:
    """Return path to one valid test fundus image."""
    if CLEAN_MANIFEST_CSV.exists():
        df = pd.read_csv(CLEAN_MANIFEST_CSV)
        test_rows = df[df["orig_split"] == "test"]
        if not test_rows.empty:
            p = Path(test_rows.iloc[0]["filepath"])
            if p.exists():
                return p
    for sub in (DATASET_DIR / "test").iterdir():
        if sub.is_dir() and sub.name.lower() != "other":
            files = sorted(f for f in sub.iterdir() if f.is_file())
            if files:
                return files[0]
    raise FileNotFoundError("No sample test image found.")


def _encode_bgr(img_bgr: np.ndarray, ext: str = ".jpg", params: list[int] | None = None) -> bytes:
    success, buf = cv2.imencode(ext, img_bgr, params or [])
    assert success, f"Failed to encode image as {ext}"
    return buf.tobytes()


@pytest.fixture(scope="module")
def sample_bgr() -> np.ndarray:
    path = _get_sample_test_image_path()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None
    return img


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


# ==============================================================================
# 1. Unit Tests for FundusImage Class
# ==============================================================================
def test_fundus_image_attributes_and_hash(sample_bgr: np.ndarray) -> None:
    jpg_bytes = _encode_bgr(sample_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    fimg = FundusImage(jpg_bytes, filename="sample_fundus.jpg")

    assert fimg.format == "JPEG"
    assert fimg.lebarPx == sample_bgr.shape[1]
    assert fimg.tinggiPx == sample_bgr.shape[0]
    assert fimg.channels == 3
    assert len(fimg.sha256) == 64
    assert fimg.computeHash() == fimg.sha256
    assert fimg.validateFormat() is True


def test_validate_format_default_vs_explicit_allowed_formats(sample_bgr: np.ndarray) -> None:
    webp_bytes = _encode_bgr(sample_bgr, ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), 80])
    fimg_webp = FundusImage(webp_bytes, filename="eye.webp")
    assert fimg_webp.format == "WEBP"

    # Default allowed_formats={"JPEG", "PNG"} must reject WEBP
    with pytest.raises(ImageValidationError):
        fimg_webp.validateFormat()

    # Explicit robustness mode allows WEBP and BMP
    assert fimg_webp.validateFormat(allowed_formats={"JPEG", "PNG", "WEBP", "BMP"}) is True


def test_strip_exif_removes_camera_and_gps_metadata(sample_bgr: np.ndarray) -> None:
    rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    exif = pil_img.getexif()
    exif[0x010F] = "Canon Medical"          # Make
    exif[0x0110] = "CR-2 Retinal Camera"    # Model
    exif[0x0131] = "Patient_NIK_999999"     # Software / sensitive tag

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    raw_with_exif = buf.getvalue()

    fimg = FundusImage(raw_with_exif, filename="patient_with_exif.jpg")
    assert fimg.has_exif() is True
    hash_before = fimg.sha256

    sanitized_bytes = fimg.stripExif()
    assert fimg.has_exif() is False
    assert FundusImage(sanitized_bytes).has_exif() is False
    assert fimg.sha256 != hash_before
    assert fimg.lebarPx == sample_bgr.shape[1]
    assert fimg.tinggiPx == sample_bgr.shape[0]
    assert fimg.validateFormat() is True


def test_channel_robustness_rgba_and_grayscale(sample_bgr: np.ndarray) -> None:
    # 4-channel BGRA PNG
    alpha = np.full((sample_bgr.shape[0], sample_bgr.shape[1], 1), 180, dtype=np.uint8)
    bgra = np.concatenate([sample_bgr, alpha], axis=2)
    png_rgba_bytes = _encode_bgr(bgra, ".png")

    fimg_rgba = FundusImage(png_rgba_bytes, filename="rgba_eye.png")
    assert fimg_rgba.format == "PNG"
    assert fimg_rgba.channels == 4
    assert fimg_rgba.validateFormat() is True
    bgr_from_rgba = fimg_rgba.to_bgr_array()
    assert bgr_from_rgba.shape == (sample_bgr.shape[0], sample_bgr.shape[1], 3)
    prep_rgba = preprocess_image(bgr_from_rgba, return_rgb=True)
    assert prep_rgba.shape == (224, 224, 3)

    # 1-channel Grayscale PNG
    gray = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)
    png_gray_bytes = _encode_bgr(gray, ".png")

    fimg_gray = FundusImage(png_gray_bytes, filename="gray_eye.png")
    assert fimg_gray.format == "PNG"
    assert fimg_gray.channels == 1
    assert fimg_gray.validateFormat() is True
    bgr_from_gray = fimg_gray.to_bgr_array()
    assert bgr_from_gray.shape == (sample_bgr.shape[0], sample_bgr.shape[1], 3)
    prep_gray = preprocess_image(bgr_from_gray, return_rgb=True)
    assert prep_gray.shape == (224, 224, 3)


def test_resolution_boundary_validation(sample_bgr: np.ndarray) -> None:
    # Below 224x224 -> must fail
    tiny_bgr = cv2.resize(sample_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    tiny_bytes = _encode_bgr(tiny_bgr, ".jpg")
    fimg_tiny = FundusImage(tiny_bytes, filename="tiny.jpg")
    assert fimg_tiny.lebarPx == 128 and fimg_tiny.tinggiPx == 128
    with pytest.raises(ImageValidationError, match="below the minimum required resolution"):
        fimg_tiny.validateFormat()

    # Exactly 224x224 -> must pass
    exact_bgr = cv2.resize(sample_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    exact_bytes = _encode_bgr(exact_bgr, ".jpg")
    fimg_exact = FundusImage(exact_bytes, filename="exact224.jpg")
    assert fimg_exact.validateFormat() is True


def test_negative_validation_empty_truncated_and_spoofed(sample_bgr: np.ndarray) -> None:
    # 1. Empty 0-byte payload
    with pytest.raises(ImageValidationError, match="empty"):
        FundusImage(b"", filename="empty.jpg").validateFormat()

    # 2. Truncated JPEG (cut at 50% of byte stream)
    valid_jpg = _encode_bgr(sample_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    truncated_jpg = valid_jpg[: len(valid_jpg) // 2]
    with pytest.raises(ImageValidationError):
        FundusImage(truncated_jpg, filename="truncated.jpg").validateFormat()

    # 3. Magic-bytes spoofing (PDF header disguised as mata_pasien.jpg)
    fake_pdf_bytes = b"%PDF-1.5\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    with pytest.raises(ImageValidationError, match="spoofed|Invalid"):
        FundusImage(fake_pdf_bytes, filename="mata_pasien.jpg").validateFormat()

    # 4. Extension vs magic-bytes mismatch (PNG bytes named .jpg)
    valid_png = _encode_bgr(sample_bgr, ".png")
    with pytest.raises(ImageValidationError, match="does not match"):
        FundusImage(valid_png, filename="spoofed_png.jpg").validateFormat()


# ==============================================================================
# 2. FastAPI Endpoint Integration Tests (POST /api/v1/predict)
# ==============================================================================
def test_api_png_lossless_vs_jpg_q95_consistency(api_client: TestClient, sample_bgr: np.ndarray) -> None:
    png_bytes = _encode_bgr(sample_bgr, ".png")
    jpg_bytes = _encode_bgr(sample_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    resp_png = api_client.post(
        "/api/v1/predict", files={"file": ("eye.png", png_bytes, "image/png")}
    )
    resp_jpg = api_client.post(
        "/api/v1/predict", files={"file": ("eye.jpg", jpg_bytes, "image/jpeg")}
    )

    assert resp_png.status_code == 200
    assert resp_jpg.status_code == 200
    data_png = resp_png.json()
    data_jpg = resp_jpg.json()

    assert data_png["confidence_threshold"] == CONFIDENCE_THRESHOLD == 0.90
    assert data_png["predicted_class"] == data_jpg["predicted_class"]
    assert abs(data_png["confidence"] - data_jpg["confidence"]) < 0.05


def test_api_rgba_and_grayscale_uploads(api_client: TestClient, sample_bgr: np.ndarray) -> None:
    alpha = np.full((sample_bgr.shape[0], sample_bgr.shape[1], 1), 200, dtype=np.uint8)
    bgra = np.concatenate([sample_bgr, alpha], axis=2)
    rgba_bytes = _encode_bgr(bgra, ".png")

    gray = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)
    gray_bytes = _encode_bgr(gray, ".png")

    for name, payload in [("rgba.png", rgba_bytes), ("gray.png", gray_bytes)]:
        resp = api_client.post(
            "/api/v1/predict", files={"file": (name, payload, "image/png")}
        )
        assert resp.status_code == 200, f"Failed for {name}: {resp.text}"
        body = resp.json()
        assert body["predicted_class"] in CLASSES
        assert body["gradcam_base64"].startswith("data:image/jpeg;base64,")


def test_api_security_and_negative_inputs(api_client: TestClient, sample_bgr: np.ndarray) -> None:
    # 1. Empty 0-byte file
    r_empty = api_client.post(
        "/api/v1/predict", files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert r_empty.status_code == 400

    # 2. Truncated JPEG bytes
    valid_jpg = _encode_bgr(sample_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    r_trunc = api_client.post(
        "/api/v1/predict",
        files={"file": ("truncated.jpg", valid_jpg[: len(valid_jpg) // 2], "image/jpeg")},
    )
    assert r_trunc.status_code == 400

    # 3. Fake extension / magic-bytes spoofing (PDF/text renamed to mata_pasien.jpg)
    r_spoof = api_client.post(
        "/api/v1/predict",
        files={"file": ("mata_pasien.jpg", b"%PDF-1.4 fake document content", "image/jpeg")},
    )
    assert r_spoof.status_code == 400

    # 4. Below minimum 224x224 resolution (112x112)
    small_bgr = cv2.resize(sample_bgr, (112, 112), interpolation=cv2.INTER_AREA)
    small_jpg = _encode_bgr(small_bgr, ".jpg")
    r_small = api_client.post(
        "/api/v1/predict",
        files={"file": ("small_112.jpg", small_jpg, "image/jpeg")},
    )
    assert r_small.status_code == 400

    # 5. WebP rejected on production endpoint (default allowed_formats={"JPEG", "PNG"})
    webp_bytes = _encode_bgr(sample_bgr, ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), 80])
    r_webp = api_client.post(
        "/api/v1/predict",
        files={"file": ("test.webp", webp_bytes, "image/webp")},
    )
    assert r_webp.status_code == 400
