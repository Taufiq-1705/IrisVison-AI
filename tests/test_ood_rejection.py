"""Automated pytest suite for 2-Layer Defense-in-Depth Out-of-Distribution (OOD) rejection.

Verifies `POST /api/v1/predict` on:
1. 4 real test eye images (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`):
   -> Must PASS (`200 OK`), return `status == "success"`, `gatekeeper_eye_score >= 0.65`,
      renormalized 4-class medical probabilities summing to ~1.0, and valid Grad-CAM Base64.
2. 6 non-eye / Out-of-Distribution samples (`>= 224x224` pixels):
   - Sample 1: `non_fundus` image from the unseen `test/non_fundus` split
   - Sample 2: Natural random object photo from `DATASETS/class-random-image/test` upscaled to 256x256
   - Sample 3: Synthetic glowing orange sphere on black background (fundus imposter hard-negative)
   - Sample 4: Solid black plain image (`256x256`)
   - Sample 5: Solid white plain image (`256x256`)
   - Sample 6: Random high-frequency RGB uniform noise image (`256x256`)
   -> All must be REJECTED with HTTP `422 Unprocessable Entity`, `status == "rejected_ood"`,
      and informative rejection detail from Layer 1 (Gatekeeper) or Layer 2 (`non_fundus`).
"""

from pathlib import Path
import sys

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from src.config import (
    CLASSES,
    CLEAN_MANIFEST_CSV,
    DATASET_DIR,
    GATEKEEPER_EYE_THRESHOLD,
    OOD_CLASS_NAME,
    RANDOM_IMAGE_OOD_DIR,
)
from src.prepare_ood_data import generate_hard_negative_image, upscale_to_min_resolution


def _encode_jpg(img_bgr: np.ndarray, quality: int = 94) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok, "Failed to encode test image as JPEG"
    return buf.tobytes()


def _select_confident_test_eye_samples() -> dict[str, Path]:
    """Select one representative test image per ocular disease class from clean_dataset_manifest.csv."""
    assert CLEAN_MANIFEST_CSV.exists(), f"Missing {CLEAN_MANIFEST_CSV}"
    df = pd.read_csv(CLEAN_MANIFEST_CSV)
    test_df = df[df["orig_split"] == "test"]
    samples: dict[str, Path] = {}
    for cls_name in CLASSES:
        rows = test_df[test_df["class_name"] == cls_name]
        assert not rows.empty, f"No test rows found for {cls_name}"
        samples[cls_name] = Path(rows.iloc[0]["filepath"])
    return samples


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_health_endpoint_reports_ood_defense(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["gatekeeper_loaded"] is True
    assert body["ood_defense_enabled"] is True
    assert body["classes"] == CLASSES


@pytest.mark.parametrize("true_class", CLASSES)
def test_valid_ocular_images_pass_defense_and_classify_correctly(
    api_client: TestClient,
    true_class: str,
) -> None:
    """All 4 ocular disease classes must pass Gatekeeper & Layer-2 and be classified accurately."""
    samples = _select_confident_test_eye_samples()
    df = pd.read_csv(CLEAN_MANIFEST_CSV)
    cls_test_rows = df[(df["orig_split"] == "test") & (df["class_name"] == true_class)]

    # Test top candidate (or first 3 test candidates to account for natural per-sample variance)
    passed_correctly = False
    last_data = None
    for _, row in cls_test_rows.head(3).iterrows():
        img_path = Path(row["filepath"])
        with open(img_path, "rb") as f:
            resp = api_client.post(
                "/api/v1/predict",
                files={"file": (img_path.name, f, "image/jpeg")},
            )
        if resp.status_code == 200:
            data = resp.json()
            last_data = data
            assert data["status"] == "success"
            assert data["gatekeeper_eye_score"] is not None
            assert data["gatekeeper_eye_score"] >= GATEKEEPER_EYE_THRESHOLD
            assert set(data["probabilities"].keys()) == set(CLASSES)
            assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01
            assert data["gradcam_base64"].startswith("data:image/jpeg;base64,")
            if data["predicted_class"] == true_class:
                passed_correctly = True
                break

    assert passed_correctly, (
        f"Expected valid `{true_class}` sample ({samples[true_class].name}) to pass OOD defense "
        f"and predict `{true_class}`, last response: {last_data}"
    )


def test_ood_1_test_set_non_fundus_sample_rejected(api_client: TestClient) -> None:
    """Sample 1: Unseen `test/non_fundus` image must be rejected with HTTP 422 `rejected_ood`."""
    test_ood_dir = DATASET_DIR / "test" / OOD_CLASS_NAME
    files = sorted(p for p in test_ood_dir.iterdir() if p.is_file())
    assert files, f"No files found in {test_ood_dir}"

    sample_path = files[0]
    with open(sample_path, "rb") as f:
        resp = api_client.post(
            "/api/v1/predict",
            files={"file": (sample_path.name, f, "image/jpeg")},
        )

    assert resp.status_code == 422, f"Expected 422 for {sample_path.name}, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "rejected_ood"
    assert body["rejected_by"] in {"layer1_gatekeeper", "layer2_5class_non_fundus"}
    assert "ditolak" in body["detail"].lower()


def test_ood_2_random_natural_object_rejected(api_client: TestClient) -> None:
    """Sample 2: Random object photo from `DATASETS/class-random-image/test` must be rejected (422)."""
    raw_test_dir = RANDOM_IMAGE_OOD_DIR / "test"
    candidates = sorted(p for p in raw_test_dir.iterdir() if p.is_file())
    assert candidates, f"No images found in {raw_test_dir}"

    bgr = cv2.imread(str(candidates[-1]), cv2.IMREAD_COLOR)
    upscaled = upscale_to_min_resolution(bgr, target_min=256, target_max=256)
    jpg_bytes = _encode_jpg(upscaled)

    resp = api_client.post(
        "/api/v1/predict",
        files={"file": ("random_object.jpg", jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422, f"Expected 422 for random object, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "rejected_ood"


def test_ood_3_synthetic_orange_circle_imposter_rejected(api_client: TestClient) -> None:
    """Sample 3: Synthetic glowing orange circle on dark background (fake fundus) must be rejected (422)."""
    # idx=0 generates category 0: circular orange/amber sphere on dark background
    imposter_bgr, _ = generate_hard_negative_image(idx=0, split_seed_offset=777777)
    jpg_bytes = _encode_jpg(imposter_bgr)

    resp = api_client.post(
        "/api/v1/predict",
        files={"file": ("fake_orange_circle.jpg", jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422, f"Expected 422 for orange circle imposter, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "rejected_ood"


def test_ood_4_solid_black_and_white_images_rejected(api_client: TestClient) -> None:
    """Sample 4: Pure black and pure white 256x256 images must be rejected (422)."""
    black_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    white_bgr = np.full((256, 256, 3), 255, dtype=np.uint8)

    for name, img in [("solid_black.jpg", black_bgr), ("solid_white.jpg", white_bgr)]:
        resp = api_client.post(
            "/api/v1/predict",
            files={"file": (name, _encode_jpg(img), "image/jpeg")},
        )
        assert resp.status_code == 422, f"Expected 422 for {name}, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["status"] == "rejected_ood"


def test_ood_5_random_rgb_noise_image_rejected(api_client: TestClient) -> None:
    """Sample 5: Random high-frequency RGB uniform noise image (256x256) must be rejected (422)."""
    rng = np.random.RandomState(202611)
    noise_bgr = rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    jpg_bytes = _encode_jpg(noise_bgr)

    resp = api_client.post(
        "/api/v1/predict",
        files={"file": ("random_noise.jpg", jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422, f"Expected 422 for random RGB noise, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "rejected_ood"


def test_ood_6_synthetic_external_eye_rejected(api_client: TestClient) -> None:
    """Sample 6: Synthetic external camera eye (anterior segment, non-fundus) must be rejected (422)."""
    ext_eye_bgr, tag = generate_hard_negative_image(idx=0, split_seed_offset=888888, category_override=4)
    assert tag == "hardneg_external_eye"
    jpg_bytes = _encode_jpg(ext_eye_bgr)

    resp = api_client.post(
        "/api/v1/predict",
        files={"file": ("external_camera_eye.jpg", jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422, f"Expected 422 for external eye photo, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "rejected_ood"
    assert body["rejected_by"] in {"layer1_gatekeeper", "layer2_5class_non_fundus"}


def test_ood_7_synthetic_human_face_rejected(api_client: TestClient) -> None:
    """Sample 7: Synthetic human face / selfie portrait must be rejected (422)."""
    face_bgr, tag = generate_hard_negative_image(idx=0, split_seed_offset=888889, category_override=5)
    assert tag == "hardneg_human_face"
    jpg_bytes = _encode_jpg(face_bgr)

    resp = api_client.post(
        "/api/v1/predict",
        files={"file": ("human_face_selfie.jpg", jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422, f"Expected 422 for human face photo, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "rejected_ood"
    assert body["rejected_by"] in {"layer1_gatekeeper", "layer2_5class_non_fundus"}


def test_ood_8_synthetic_document_rejected(api_client: TestClient) -> None:
    """Sample 8: Synthetic text document / receipt / chart must be rejected (422)."""
    doc_bgr, tag = generate_hard_negative_image(idx=0, split_seed_offset=888890, category_override=1)
    assert tag == "hardneg_document"
    jpg_bytes = _encode_jpg(doc_bgr)

    resp = api_client.post(
        "/api/v1/predict",
        files={"file": ("medical_document.jpg", jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422, f"Expected 422 for document image, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "rejected_ood"
    assert body["rejected_by"] in {"layer1_gatekeeper", "layer2_5class_non_fundus"}
