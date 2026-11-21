"""Automated API test suite for AI Sentinel (Iris Vision) FastAPI backend.

Tests:
1. GET /health (server status, model metadata, GPU/CPU compute state, 80% threshold)
2. GET / (interactive HTML diagnostic dashboard)
3. POST /api/v1/predict across 1 test sample from each of the 4 classes:
   (cataract, diabetic_retinopathy, glaucoma, normal)
   validating Softmax probabilities, Laplacian blur check, 80% threshold logic,
   and Base64 Grad-CAM output.
4. Negative tests: empty upload, invalid extension, and corrupt image bytes.
"""

import base64
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from src.config import CLASSES, CLEAN_MANIFEST_CSV, CONFIDENCE_THRESHOLD, DATASET_DIR


def get_one_test_image_per_class() -> dict[str, Path]:
    """Select one unseen test image for each of the 4 ocular disease classes."""
    samples: dict[str, Path] = {}

    if CLEAN_MANIFEST_CSV.exists():
        df = pd.read_csv(CLEAN_MANIFEST_CSV)
        test_df = df[df["orig_split"] == "test"]
        for cls_name in CLASSES:
            cls_rows = test_df[test_df["class_name"] == cls_name]
            if not cls_rows.empty:
                samples[cls_name] = Path(cls_rows.iloc[0]["filepath"])

    # Fallback directly to directory scan if needed
    for cls_name in CLASSES:
        if cls_name not in samples:
            cls_dir = DATASET_DIR / "test" / cls_name
            files = sorted(p for p in cls_dir.iterdir() if p.is_file())
            if not files:
                raise FileNotFoundError(f"No test images found for class: {cls_name}")
            samples[cls_name] = files[0]

    return samples


def run_api_tests() -> None:
    """Execute end-to-end API tests using FastAPI TestClient."""
    print("=" * 84)
    print("AI SENTINEL (IRIS VISION) - FASTAPI ENDPOINT TEST SUITE")
    print("=" * 84)

    with TestClient(app) as client:
        # 1. Health Endpoint
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Unexpected status: {health_resp.status_code}"
        health_data = health_resp.json()
        print(f"[PASS] GET /health -> {health_data}")
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True
        assert health_data["confidence_threshold"] == CONFIDENCE_THRESHOLD
        assert health_data["classes"] == CLASSES

        # 2. Dashboard Endpoint
        ui_resp = client.get("/")
        assert ui_resp.status_code == 200
        assert "AI Sentinel" in ui_resp.text
        print("[PASS] GET / -> HTML Diagnostic Dashboard served cleanly.")

        # 3. Predict Endpoint across 4 classes
        samples = get_one_test_image_per_class()
        print(
            f"\n--- Testing POST /api/v1/predict "
            f"(1 Sample per Class, Threshold={int(CONFIDENCE_THRESHOLD * 100)}%) ---"
        )
        header = (
            f"{'True Class':<22} | {'Filename':<18} | {'Predicted':<22} | "
            f"{'Conf':>7} | {'Status':<10} | {'BlurVar':>8} | {'Time(ms)':>8}"
        )
        print(header)
        print("-" * len(header))

        for true_cls in CLASSES:
            img_path = samples[true_cls]
            with open(img_path, "rb") as f:
                resp = client.post(
                    "/api/v1/predict",
                    files={"file": (img_path.name, f, "image/jpeg")},
                )

            assert resp.status_code == 200, f"Failed for {true_cls}: {resp.text}"
            data = resp.json()

            # Structural & Mathematical Assertions
            assert data["status"] == "success"
            assert data["predicted_class"] in CLASSES
            assert 0.0 <= data["confidence"] <= 1.0
            assert data["confidence_threshold"] == CONFIDENCE_THRESHOLD
            assert data["is_confident"] == (data["confidence"] >= CONFIDENCE_THRESHOLD)
            assert data["diagnosis_status"] == ("confident" if data["is_confident"] else "uncertain")

            probs = data["probabilities"]
            assert set(probs.keys()) == set(CLASSES)
            prob_sum = sum(probs.values())
            assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum mismatch: {prob_sum}"

            blur = data["blur_assessment"]
            assert "laplacian_variance" in blur and blur["laplacian_variance"] > 0

            b64_str = data["gradcam_base64"]
            assert b64_str.startswith("data:image/jpeg;base64,")
            decoded_bytes = base64.b64decode(b64_str.split(",", 1)[1])
            assert len(decoded_bytes) > 1000, "Decoded Grad-CAM image too small"

            print(
                f"{true_cls:<22} | {img_path.name:<18} | {data['predicted_class']:<22} | "
                f"{data['confidence']*100:6.2f}% | {data['diagnosis_status']:<10} | "
                f"{blur['laplacian_variance']:8.1f} | {data['inference_time_ms']:8.2f}"
            )

        # 4. Negative Error Handling Tests
        print("\n--- Testing Error Handling (Invalid & Corrupt Inputs) ---")
        empty_resp = client.post(
            "/api/v1/predict", files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert empty_resp.status_code == 400
        print(f"[PASS] Empty file rejected (HTTP {empty_resp.status_code}): {empty_resp.json()['detail']}")

        txt_resp = client.post(
            "/api/v1/predict", files={"file": ("notes.txt", b"hello world", "text/plain")}
        )
        assert txt_resp.status_code == 400
        print(f"[PASS] Non-image file rejected (HTTP {txt_resp.status_code}): {txt_resp.json()['detail']}")

        corrupt_resp = client.post(
            "/api/v1/predict", files={"file": ("corrupt.jpg", b"not_valid_jpeg_bytes", "image/jpeg")}
        )
        assert corrupt_resp.status_code == 400
        print(f"[PASS] Corrupt JPEG rejected (HTTP {corrupt_resp.status_code}): {corrupt_resp.json()['detail']}")

    print("\n" + "=" * 84)
    print("[SUCCESS] All AI Sentinel FastAPI tests passed!")
    print("=" * 84)


if __name__ == "__main__":
    run_api_tests()

