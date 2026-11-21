"""White-box unit tests for IrisVision core internal logic.

Focuses on statement, branch/decision, and exception path coverage across:
1. `src/preprocess.py`: border brightness, safe_crop_fundus_roi guards & ROI cropping,
   LAB CLAHE, and RGB/BGR output branching.
2. `src/dataset.py`: balance_classes_df strategies and error branches.
3. `src/image_validator.py`: detect_magic_format magic-byte signature branches.
4. `app/main.py`: assess_image_blur threshold branching and health_check degraded mode.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app_state, assess_image_blur, health_check
from src.config import BLUR_LAPLACIAN_THRESHOLD, IMAGE_SIZE
from src.dataset import EyeDiseaseDataset, balance_classes_df
from src.image_validator import detect_magic_format
from src.preprocess import (
    apply_clahe_lab,
    compute_border_brightness,
    preprocess_image,
    safe_crop_fundus_roi,
)


# ==============================================================================
# 1. White-Box Tests: src/preprocess.py
# ==============================================================================
def test_compute_border_brightness_branches() -> None:
    """Test empty/None guard, 2D grayscale, and 3D BGR branches."""
    assert compute_border_brightness(None) == 0.0
    assert compute_border_brightness(np.array([], dtype=np.uint8)) == 0.0

    gray_img = np.full((100, 100), 40, dtype=np.uint8)
    assert compute_border_brightness(gray_img) == pytest.approx(40.0)

    bgr_img = np.full((100, 100, 3), 20, dtype=np.uint8)
    assert compute_border_brightness(bgr_img) == pytest.approx(20.0)


@pytest.mark.parametrize("bad_input", [None, np.array([], dtype=np.uint8)])
def test_safe_crop_fundus_roi_invalid_input_raises(bad_input) -> None:
    """Empty or None input must raise ValueError."""
    with pytest.raises(ValueError, match="empty or None"):
        safe_crop_fundus_roi(bad_input)


def test_safe_crop_fundus_roi_channel_normalization() -> None:
    """2D grayscale and 4-channel BGRA inputs must be normalized to 3-channel BGR."""
    gray = np.full((100, 100), 128, dtype=np.uint8)
    out_gray = safe_crop_fundus_roi(gray)
    assert out_gray.shape == (100, 100, 3)

    bgra = np.full((100, 100, 4), 128, dtype=np.uint8)
    out_bgra = safe_crop_fundus_roi(bgra)
    assert out_bgra.shape == (100, 100, 3)


def test_safe_crop_fundus_roi_uniform_cataract_fundus_cropping() -> None:
    """All ocular classes (including cataract) are 100% fundus and must be cropped uniformly."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 60, (0, 200, 220), -1)

    out_with_class = safe_crop_fundus_roi(img, class_name=" Cataract ")
    out_inference = safe_crop_fundus_roi(img, class_name=None)

    assert out_with_class.shape[0] < 200 and out_with_class.shape[1] < 200
    assert np.array_equal(out_with_class, out_inference)


def test_preprocess_parity_dataset_vs_api_for_cataract_fundus(tmp_path: Path) -> None:
    """Verify training/eval (`EyeDiseaseDataset`) and API (`app/main.py`) produce identical RGB for cataract fundus."""
    img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img_bgr, (150, 150), 95, (30, 85, 210), -1)
    cv2.circle(img_bgr, (125, 145), 18, (80, 190, 245), -1)

    img_path = tmp_path / "cataract_fundus_sample.png"
    assert cv2.imwrite(str(img_path), img_bgr)

    df = pd.DataFrame([{"filepath": str(img_path), "class_name": "cataract", "label_idx": 0}])

    # Dataset path with pass_class_name_to_roi=False (default) and True (legacy caller)
    ds_default = EyeDiseaseDataset(df, use_preprocess=True, pass_class_name_to_roi=False, preload_in_memory=False)
    ds_with_cls = EyeDiseaseDataset(df, use_preprocess=True, pass_class_name_to_roi=True, preload_in_memory=False)

    rgb_ds_default = ds_default._load_single_rgb((str(img_path), None))
    rgb_ds_with_cls = ds_with_cls._load_single_rgb((str(img_path), "cataract"))

    # Exact call pattern used in app/main.py during POST /api/v1/predict
    loaded_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rgb_api = preprocess_image(loaded_bgr, class_name=None, target_size=IMAGE_SIZE, return_rgb=True)

    assert rgb_api.shape == (*IMAGE_SIZE, 3)
    assert np.array_equal(rgb_ds_default, rgb_api)
    assert np.array_equal(rgb_ds_with_cls, rgb_api)


def test_safe_crop_fundus_roi_bright_border_guard() -> None:
    """Layer 2 guard: bright perimeter skips cropping even when class_name is None."""
    img = np.full((200, 200, 3), 80, dtype=np.uint8)
    out = safe_crop_fundus_roi(img, class_name=None)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_safe_crop_fundus_roi_no_contours_and_tiny_roi_guards() -> None:
    """All-black image (no contours) and tiny ROI (<32px) must return original image."""
    all_black = np.zeros((200, 200, 3), dtype=np.uint8)
    out_black = safe_crop_fundus_roi(all_black)
    assert out_black.shape == all_black.shape

    tiny_spot = np.zeros((200, 200, 3), dtype=np.uint8)
    tiny_spot[95:105, 95:105] = (0, 0, 255)  # 10x10 spot (< 32px)
    out_tiny = safe_crop_fundus_roi(tiny_spot)
    assert out_tiny.shape == tiny_spot.shape


def test_safe_crop_fundus_roi_valid_crop() -> None:
    """Dark border + large central retinal circle must crop down to ROI + padding."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (20, 40, 200), -1)

    cropped = safe_crop_fundus_roi(img, padding=2)
    assert cropped.shape[0] < 300
    assert cropped.shape[1] < 300
    assert cropped.shape[2] == 3


def test_apply_clahe_lab_and_preprocess_color_branches() -> None:
    """Test CLAHE ValueError guard and preprocess_image return_rgb True/False."""
    with pytest.raises(ValueError, match="empty or None"):
        apply_clahe_lab(np.array([], dtype=np.uint8))

    # Distinct B vs R values to verify BGR -> RGB channel swap
    img_bgr = np.full((240, 240, 3), (200, 100, 50), dtype=np.uint8)
    out_rgb = preprocess_image(img_bgr, return_rgb=True)
    out_bgr = preprocess_image(img_bgr, return_rgb=False)

    assert out_rgb.shape == (*IMAGE_SIZE, 3)
    assert out_bgr.shape == (*IMAGE_SIZE, 3)
    assert np.array_equal(out_rgb, cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))


# ==============================================================================
# 2. White-Box Tests: src/dataset.py
# ==============================================================================
def _make_imbalanced_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filepath": [f"img_{i}.jpg" for i in range(6)],
            "class_name": ["cataract"] * 4 + ["normal"] * 2,
        }
    )


def test_balance_classes_df_empty_and_strategies() -> None:
    """Test empty df, oversample, downsample, and target strategies."""
    empty_df = pd.DataFrame(columns=["filepath", "class_name"])
    assert balance_classes_df(empty_df).empty

    df = _make_imbalanced_df()

    over_df = balance_classes_df(df, strategy="oversample", seed=42)
    assert over_df["class_name"].value_counts().to_dict() == {"cataract": 4, "normal": 4}

    down_df = balance_classes_df(df, strategy="downsample", seed=42)
    assert down_df["class_name"].value_counts().to_dict() == {"cataract": 2, "normal": 2}

    target_df = balance_classes_df(df, strategy="target", target_count=3, seed=42)
    assert target_df["class_name"].value_counts().to_dict() == {"cataract": 3, "normal": 3}


@pytest.mark.parametrize(
    "strategy,target_count",
    [
        ("target", None),
        ("target", 0),
        ("target", -5),
        ("invalid_strategy", None),
    ],
)
def test_balance_classes_df_invalid_args_raise(strategy: str, target_count) -> None:
    """Invalid target_count or unknown strategy must raise ValueError."""
    df = _make_imbalanced_df()
    with pytest.raises(ValueError):
        balance_classes_df(df, strategy=strategy, target_count=target_count)


# ==============================================================================
# 3. White-Box Tests: src/image_validator.py
# ==============================================================================
@pytest.mark.parametrize(
    "payload,expected",
    [
        (b"", "UNKNOWN"),
        (b"\xff\xd8", "UNKNOWN"),  # < 4 bytes
        (b"\xff\xd8\xff\xe0rest", "JPEG"),
        (b"\x89PNG\r\n\x1a\nrest", "PNG"),
        (b"RIFF1234WEBPrest", "WEBP"),
        (b"BM123456", "BMP"),
        (b"%PDF-1.4\n", "UNKNOWN"),
    ],
)
def test_detect_magic_format_all_branches(payload: bytes, expected: str) -> None:
    """Every magic-byte signature branch must return the expected container type."""
    assert detect_magic_format(payload) == expected


# ==============================================================================
# 4. White-Box Tests: app/main.py
# ==============================================================================
def test_assess_image_blur_branches() -> None:
    """Test blurry (variance < threshold) vs sharp (variance >= threshold) paths."""
    flat_img = np.full((224, 224, 3), 128, dtype=np.uint8)
    blur_res = assess_image_blur(flat_img)
    assert blur_res.laplacian_variance < BLUR_LAPLACIAN_THRESHOLD
    assert blur_res.is_blurry is True
    assert blur_res.warning is not None

    # High-frequency checkerboard pattern yields high Laplacian variance
    sharp_img = np.zeros((224, 224, 3), dtype=np.uint8)
    sharp_img[::2, ::2] = 255
    sharp_img[1::2, 1::2] = 255
    sharp_res = assess_image_blur(sharp_img)
    assert sharp_res.laplacian_variance >= BLUR_LAPLACIAN_THRESHOLD
    assert sharp_res.is_blurry is False
    assert sharp_res.warning is None


def test_health_check_degraded_branch() -> None:
    """When app_state['model'] is None, health_check must report degraded status."""
    orig_model = app_state["model"]
    orig_device = app_state["device"]
    try:
        app_state["model"] = None
        app_state["device"] = None
        resp = health_check()
        assert resp.status == "degraded"
        assert resp.model_loaded is False
        assert resp.device == "unknown"
    finally:
        app_state["model"] = orig_model
        app_state["device"] = orig_device
