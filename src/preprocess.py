"""Image preprocessing utilities for Iris Vision.

Provides:
1. Border brightness computation to detect circular fundus images with dark margins
   uniformly across all ocular classes (100% fundus dataset).
2. `safe_crop_fundus_roi()` to remove black margins around fundus images using
   automatic border brightness thresholding (identical during training & inference).
3. Resize to target resolution (224x224) and local contrast enhancement via
   CLAHE in LAB color space.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    FUNDUS_BORDER_BRIGHTNESS_THRESHOLD,
    FUNDUS_BORDER_RATIO,
    FUNDUS_ROI_THRESHOLD,
    IMAGE_SIZE,
    MIN_ROI_AREA_RATIO,
)


def compute_border_brightness(
    img: np.ndarray,
    border_ratio: float = FUNDUS_BORDER_RATIO,
) -> float:
    """Compute mean luminance along the outer perimeter strips of an image.

    Fundus images typically exhibit near-black borders (mean ~ 0-15), whereas
    external slit-lamp / anterior-segment photos (cataract) fill the frame
    with significantly brighter borders.
    """
    if img is None or img.size == 0:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    bh = max(1, int(round(h * border_ratio)))
    bw = max(1, int(round(w * border_ratio)))

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    return float(gray[border_mask].mean())


def safe_crop_fundus_roi(
    img: np.ndarray,
    class_name: Optional[str] = None,
    border_ratio: float = FUNDUS_BORDER_RATIO,
    border_brightness_threshold: float = FUNDUS_BORDER_BRIGHTNESS_THRESHOLD,
    roi_threshold: int = FUNDUS_ROI_THRESHOLD,
    min_roi_area_ratio: float = MIN_ROI_AREA_RATIO,
    padding: int = 2,
) -> np.ndarray:
    """Crop dark margins of fundus images uniformly across training and inference.

    All 4 ocular disease classes (including `cataract`) use retinal fundus imagery.
    Therefore `class_name` is kept only for backwards compatibility and ignored so
    training (`dataset.py`) and API inference (`app/main.py`) execute the exact same
    border-brightness check (`border_brightness > border_brightness_threshold`).
    """
    _ = class_name  # Ignored intentionally to prevent training-vs-inference skew

    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    # Ensure 3-channel BGR image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Automatic border brightness check (uniform for training, evaluation, and inference)
    border_brightness = compute_border_brightness(img, border_ratio=border_ratio)
    if border_brightness > border_brightness_threshold:
        return img.copy()

    # Fundus ROI bounding-box extraction using max color channel
    # (retinal tissue is bright in Red channel even in dark pathology regions)
    h, w = img.shape[:2]
    max_channel = np.max(img, axis=2)
    blurred = cv2.GaussianBlur(max_channel, (5, 5), 0)
    mask = (blurred > roi_threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img.copy()

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Safety checks to prevent degenerate or overly aggressive crops
    area_ratio = (bw * bh) / float(h * w)
    if bw < 32 or bh < 32 or area_ratio < min_roi_area_ratio:
        return img.copy()

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    return img[y1:y2, x1:x2].copy()


def apply_clahe_lab(
    img_bgr: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel in LAB space."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image for CLAHE is empty or None.")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def preprocess_image(
    img_bgr: np.ndarray,
    class_name: Optional[str] = None,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
    return_rgb: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline:
    1. safe_crop_fundus_roi() - trims black fundus margins uniformly via border brightness.
    2. Resize to target_size (default 224x224).
    3. LAB-space CLAHE local contrast enhancement.
    4. Optionally convert BGR -> RGB.
    """
    roi_img = safe_crop_fundus_roi(img_bgr, class_name=class_name)

    h, w = roi_img.shape[:2]
    target_w, target_h = target_size
    interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_CUBIC
    resized = cv2.resize(roi_img, (target_w, target_h), interpolation=interp)

    enhanced_bgr = apply_clahe_lab(
        resized,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    if return_rgb:
        return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return enhanced_bgr

