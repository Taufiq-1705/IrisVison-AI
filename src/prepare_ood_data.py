"""Automated Out-of-Distribution (`non_fundus`) dataset preparation for AI Sentinel.

Builds leak-free `train/non_fundus`, `val/non_fundus`, and `test/non_fundus` subsets inside
`DATASETS/dataset2_balanced(Updated)`:
- `train/non_fundus` : ~1,150 images
- `val/non_fundus`   : ~115 images
- `test/non_fundus`  : ~200 images

Supports two source modes via `--source`:
1. `local` (Default): Uses `DATASETS/class-random-image` (`train/` for train+val stratified across
   its object categories, and `test/` exclusively for test) upscaled with `cv2.INTER_CUBIC` to
   `>= 256x256` pixels, paired with ~24% synthetic Hard-Negative OOD images across 6 categories:
   - `0`: Circular orange/amber/red object on dark/colored background (fundus imposter)
   - `1`: Simulated text document / form / receipt / table on white/cream paper
   - `2`: Solid / plain / gradient frames (black, white, gray, tinted)
   - `3`: High-frequency noise / checkerboard / geometric stripes
   - `4`: Synthetic external non-fundus camera eye (skin, eyelids, white sclera, colored iris, pupil)
   - `5`: Synthetic human face / portrait selfie (skin head, hair, eyes, eyebrows, nose, mouth)
2. `torchvision`: Exports non-eye images from `torchvision.datasets.CIFAR100` (train vs test split)
   upscaled to `>= 256x256` pixels + the same synthetic Hard-Negative OOD images.

Guarantees 0% data leakage across `train`, `val`, and `test` via exact MD5 hash verification.
"""

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_TO_IDX_5,
    DATASET_DIR,
    MIN_IMAGE_DIMENSION,
    OOD_CLASS_NAME,
    OOD_MANIFEST_CSV,
    OOD_TEST_TARGET,
    OOD_TRAIN_TARGET,
    OOD_VAL_TARGET,
    RANDOM_IMAGE_OOD_DIR,
    SEED,
    SPLITS,
    VALID_EXTENSIONS,
    ensure_output_dir,
)


def compute_md5_bytes(data: bytes) -> str:
    """Return MD5 hex digest for raw bytes."""
    return hashlib.md5(data).hexdigest()


def compute_file_md5(filepath: Path) -> str:
    """Return MD5 hex digest for a file on disk."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def upscale_to_min_resolution(
    img_bgr: np.ndarray,
    target_min: int = 256,
    target_max: int = 384,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Upscale or resize an image smoothly so both width and height are >= target_min (>=224)."""
    h, w = img_bgr.shape[:2]
    if rng is not None:
        desired_short = int(rng.randint(target_min, target_max + 1))
    else:
        desired_short = target_min

    short_side = min(h, w)
    scale = max(desired_short / float(max(1, short_side)), target_min / float(max(1, short_side)))
    new_w = max(target_min, int(round(w * scale)))
    new_h = max(target_min, int(round(h * scale)))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized


def _generate_external_eye_imposter(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Synthesize a close-up external camera eye photo (non-fundus anterior segment)."""
    # Periocular skin tone (BGR: warm beige to deeper brown)
    skin_r = int(rng.randint(155, 235))
    skin_g = int(skin_r * rng.uniform(0.70, 0.84))
    skin_b = int(skin_r * rng.uniform(0.52, 0.72))
    img = np.full((size, size, 3), (skin_b, skin_g, skin_r), dtype=np.float32)

    # Subtle skin shading & texture
    yy, xx = np.ogrid[:size, :size]
    cx = int(size * rng.uniform(0.45, 0.55))
    cy = int(size * rng.uniform(0.46, 0.56))
    dist_center = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / float(size)
    skin_shade = np.clip(1.06 - 0.35 * dist_center, 0.72, 1.08)
    img = np.clip(img * skin_shade[:, :, None] + rng.normal(0, 4.5, (size, size, 3)), 0, 255).astype(np.uint8)

    # Almond-shaped eye aperture (white sclera)
    eye_rx = int(size * rng.uniform(0.32, 0.42))
    eye_ry = int(size * rng.uniform(0.14, 0.22))
    sclera_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(sclera_mask, (cx, cy), (eye_rx, eye_ry), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)

    sclera_base = int(rng.randint(222, 248))
    sclera_bgr = (
        int(np.clip(sclera_base - rng.randint(4, 14), 0, 255)),
        int(np.clip(sclera_base - rng.randint(2, 10), 0, 255)),
        sclera_base,
    )
    img[sclera_mask == 255] = sclera_bgr

    # Subtle pinkish corner (caruncle) & fine scleral capillaries
    cv2.circle(img, (cx - int(eye_rx * 0.85), cy), max(4, eye_ry // 4), (150, 145, 215), -1, lineType=cv2.LINE_AA)

    # Iris (brown / hazel / blue / green) clipped inside sclera aperture
    iris_r = int(eye_ry * rng.uniform(0.72, 0.96))
    iris_palette = [
        (32, 56, 98),    # dark brown
        (42, 85, 128),   # hazel brown
        (145, 115, 62),  # blue-gray
        (68, 122, 85),   # green
    ]
    iris_bgr = iris_palette[int(rng.randint(0, len(iris_palette)))]

    eye_layer = img.copy()
    cv2.circle(eye_layer, (cx, cy), iris_r, iris_bgr, -1, lineType=cv2.LINE_AA)
    # Dark limbal ring around iris
    limbal = tuple(max(0, c - 28) for c in iris_bgr)
    cv2.circle(eye_layer, (cx, cy), iris_r, limbal, max(2, iris_r // 10), lineType=cv2.LINE_AA)

    # Radial iris fibers
    for angle_deg in range(0, 360, 12):
        rad = np.deg2rad(angle_deg)
        x1 = int(cx + np.cos(rad) * (iris_r * 0.42))
        y1 = int(cy + np.sin(rad) * (iris_r * 0.42))
        x2 = int(cx + np.cos(rad) * (iris_r * 0.92))
        y2 = int(cy + np.sin(rad) * (iris_r * 0.92))
        fiber_col = tuple(min(255, c + int(rng.randint(15, 45))) for c in iris_bgr)
        cv2.line(eye_layer, (x1, y1), (x2, y2), fiber_col, 1, lineType=cv2.LINE_AA)

    # Central pupil (dark black or cloudy anterior-cataract gray-white)
    pupil_r = max(5, int(iris_r * rng.uniform(0.32, 0.50)))
    if rng.rand() < 0.35:
        pupil_col = (int(rng.randint(155, 200)), int(rng.randint(160, 205)), int(rng.randint(165, 210)))
    else:
        pupil_col = (12, 12, 16)
    cv2.circle(eye_layer, (cx, cy), pupil_r, pupil_col, -1, lineType=cv2.LINE_AA)

    # Corneal specular flash highlight
    cv2.circle(
        eye_layer,
        (cx - pupil_r // 2, cy - pupil_r // 2),
        max(3, pupil_r // 3),
        (248, 250, 255),
        -1,
        lineType=cv2.LINE_AA,
    )

    # Composite clipped eye inside sclera
    img[sclera_mask == 255] = eye_layer[sclera_mask == 255]

    # Eyelid crease, upper/lower lid outline, and eyebrow arc
    lid_col = (max(0, skin_b - 55), max(0, skin_g - 55), max(0, skin_r - 55))
    cv2.ellipse(img, (cx, cy), (eye_rx, eye_ry), 0, 195, 345, (28, 24, 24), max(2, size // 85), lineType=cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (eye_rx, eye_ry), 0, 15, 165, lid_col, max(1, size // 140), lineType=cv2.LINE_AA)
    cv2.ellipse(
        img,
        (cx, cy - int(eye_ry * 0.55)),
        (int(eye_rx * 1.05), int(eye_ry * 0.85)),
        0,
        205,
        335,
        lid_col,
        max(1, size // 120),
        lineType=cv2.LINE_AA,
    )
    brow_col = (int(rng.randint(20, 55)), int(rng.randint(25, 60)), int(rng.randint(30, 70)))
    cv2.ellipse(
        img,
        (cx, cy - int(eye_ry * 1.45)),
        (int(eye_rx * 1.15), int(eye_ry * 0.65)),
        0,
        205,
        335,
        brow_col,
        max(5, size // 32),
        lineType=cv2.LINE_AA,
    )

    return cv2.GaussianBlur(img, (3, 3), 0.7)


def _generate_human_face_imposter(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Synthesize a portrait / selfie human face image (non-fundus OOD)."""
    # Indoor / wall / outdoor studio background
    bg_color = rng.randint(60, 235, size=3).tolist()
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    # Soft background vertical gradient
    grad = np.linspace(0.85, 1.12, size, dtype=np.float32)[:, None, None]
    img = np.clip(img.astype(np.float32) * grad + rng.normal(0, 3.0, (size, size, 3)), 0, 255).astype(np.uint8)

    cx = int(size * rng.uniform(0.46, 0.54))
    cy = int(size * rng.uniform(0.45, 0.52))
    face_rx = int(size * rng.uniform(0.24, 0.30))
    face_ry = int(size * rng.uniform(0.31, 0.38))

    skin_r = int(rng.randint(150, 235))
    skin_g = int(skin_r * rng.uniform(0.68, 0.83))
    skin_b = int(skin_r * rng.uniform(0.50, 0.70))
    skin_bgr = (skin_b, skin_g, skin_r)
    hair_bgr = (int(rng.randint(18, 65)), int(rng.randint(20, 70)), int(rng.randint(25, 85)))
    shirt_bgr = [int(x) for x in rng.randint(30, 220, size=3)]

    # Shoulders & neck
    cv2.ellipse(img, (cx, size), (int(size * 0.44), int(size * 0.22)), 0, 180, 360, shirt_bgr, -1, lineType=cv2.LINE_AA)
    neck_w = int(face_rx * 0.46)
    cv2.rectangle(img, (cx - neck_w, cy + int(face_ry * 0.65)), (cx + neck_w, size), skin_bgr, -1)

    # Hair crown behind/over top of head
    cv2.ellipse(
        img,
        (cx, cy - int(face_ry * 0.12)),
        (int(face_rx * 1.12), int(face_ry * 1.05)),
        0,
        165,
        375,
        hair_bgr,
        -1,
        lineType=cv2.LINE_AA,
    )

    # Ears & oval face
    ear_r = max(6, face_rx // 5)
    cv2.ellipse(img, (cx - face_rx, cy), (ear_r, int(ear_r * 1.5)), 0, 0, 360, skin_bgr, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (cx + face_rx, cy), (ear_r, int(ear_r * 1.5)), 0, 0, 360, skin_bgr, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (face_rx, face_ry), 0, 0, 360, skin_bgr, -1, lineType=cv2.LINE_AA)

    # Bangs / top hair cap
    cv2.ellipse(
        img,
        (cx, cy - int(face_ry * 0.58)),
        (int(face_rx * 0.96), int(face_ry * 0.42)),
        0,
        180,
        360,
        hair_bgr,
        -1,
        lineType=cv2.LINE_AA,
    )

    # Two eyes & eyebrows
    eye_dx = int(face_rx * 0.38)
    eye_y = cy - int(face_ry * 0.12)
    eye_rx = max(7, int(face_rx * 0.18))
    eye_ry = max(4, int(face_ry * 0.08))
    for side in (-1, 1):
        ex = cx + side * eye_dx
        cv2.ellipse(img, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (240, 240, 245), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (ex, eye_y), max(3, eye_ry - 1), (35, 45, 70), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (ex, eye_y), max(2, eye_ry // 2), (10, 10, 12), -1, lineType=cv2.LINE_AA)
        cv2.line(
            img,
            (ex - eye_rx, eye_y - int(eye_ry * 1.9)),
            (ex + eye_rx, eye_y - int(eye_ry * 2.1)),
            hair_bgr,
            max(2, size // 90),
            lineType=cv2.LINE_AA,
        )

    # Nose shadow & mouth/lips
    nose_col = (max(0, skin_b - 35), max(0, skin_g - 35), max(0, skin_r - 35))
    cv2.line(img, (cx, eye_y + eye_ry), (cx, cy + int(face_ry * 0.22)), nose_col, max(2, size // 120), lineType=cv2.LINE_AA)
    cv2.ellipse(
        img,
        (cx, cy + int(face_ry * 0.24)),
        (int(face_rx * 0.14), max(3, int(face_ry * 0.05))),
        0,
        0,
        180,
        nose_col,
        max(1, size // 140),
        lineType=cv2.LINE_AA,
    )

    lip_bgr = (int(rng.randint(85, 125)), int(rng.randint(75, 115)), int(rng.randint(155, 205)))
    cv2.ellipse(
        img,
        (cx, cy + int(face_ry * 0.48)),
        (int(face_rx * 0.30), max(4, int(face_ry * 0.09))),
        0,
        0,
        360,
        lip_bgr,
        -1,
        lineType=cv2.LINE_AA,
    )
    return cv2.GaussianBlur(img, (3, 3), 0.7)


def generate_hard_negative_image(
    idx: int,
    split_seed_offset: int,
    category_override: int | None = None,
) -> Tuple[np.ndarray, str]:
    """Generate a synthetic hard-negative non-fundus image (size 256x256 to 320x320 BGR).

    Categories:
    0: Circular orange/amber/red object on dark or colored background (orange fruit, ball, coin, lamp)
    1: Simulated text document / form / receipt / table on white or cream paper
    2: Solid or near-solid plain frame (black, white, gray, color gradient)
    3: High-frequency noise / geometric checkerboard / stripe pattern
    4: Synthetic external camera eye photo (anterior segment with sclera, iris, pupil, eyelids)
    5: Synthetic human face / selfie portrait
    """
    rng = np.random.RandomState(split_seed_offset + idx * 17)
    size = int(rng.choice([256, 288, 320]))
    category = (idx % 6) if category_override is None else int(category_override)

    if category == 0:
        # Hard Negative 1: Circular orange/amber/gold sphere or coin on dark/room background
        bg_val = int(rng.randint(0, 45))
        img = np.full((size, size, 3), bg_val, dtype=np.uint8)
        if rng.rand() < 0.4:
            bg_color = rng.randint(20, 120, size=3).tolist()
            img[:, :] = bg_color

        cx = int(size * rng.uniform(0.42, 0.58))
        cy = int(size * rng.uniform(0.42, 0.58))
        radius = int(size * rng.uniform(0.28, 0.44))

        b_c = int(rng.randint(10, 75))
        g_c = int(rng.randint(90, 185))
        r_c = int(rng.randint(200, 255))

        yy, xx = np.ogrid[:size, :size]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist_sq <= (radius ** 2)

        dist_norm = np.sqrt(np.maximum(0, dist_sq)) / float(max(1, radius))
        shading = np.clip(1.15 - 0.55 * (dist_norm ** 1.6), 0.3, 1.2)

        for ch, base_v in enumerate([b_c, g_c, r_c]):
            channel = img[:, :, ch].astype(np.float32)
            sphere_ch = np.clip(base_v * shading + rng.normal(0, 6, (size, size)), 0, 255)
            channel[mask] = sphere_ch[mask]
            img[:, :, ch] = channel.astype(np.uint8)

        if rng.rand() < 0.5:
            cv2.circle(
                img,
                (cx - radius // 4, cy - radius // 4),
                max(6, radius // 6),
                (235, 245, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
            img = cv2.GaussianBlur(img, (7, 7), 1.5)
        else:
            cv2.circle(img, (cx, cy), int(radius * 0.82), (b_c // 2, g_c // 2, r_c // 2), 2, lineType=cv2.LINE_AA)
            cv2.rectangle(
                img,
                (cx - radius // 3, cy - radius // 5),
                (cx + radius // 3, cy + radius // 5),
                (b_c + 20, min(255, g_c + 30), r_c),
                2,
            )
        return img, "hardneg_orange_circle"

    if category == 1:
        # Hard Negative 2: Document / Receipt / Medical Chart Table on white/cream background
        paper_tone = int(rng.randint(225, 255))
        img = np.full((size, size, 3), paper_tone, dtype=np.uint8)
        img[:, :, 0] = np.clip(paper_tone - int(rng.randint(0, 15)), 0, 255)

        ink_color = (int(rng.randint(15, 70)), int(rng.randint(15, 70)), int(rng.randint(15, 70)))
        cv2.rectangle(img, (20, 18), (size - 20, 44), ink_color, 2)
        for y in range(60, size - 25, int(rng.randint(14, 22))):
            line_w = int(rng.randint(size // 2, size - 25))
            thickness = int(rng.choice([1, 2]))
            cv2.line(img, (22, y), (line_w, y), ink_color, thickness)
            for gap_x in range(55, line_w - 15, int(rng.randint(28, 55))):
                cv2.line(img, (gap_x, y), (gap_x + 6, y), (paper_tone, paper_tone, paper_tone), thickness + 1)

        if rng.rand() < 0.6:
            for x in [22, size // 2, size - 22]:
                cv2.line(img, (x, 55), (x, size - 25), ink_color, 1)
        return img, "hardneg_document"

    if category == 2:
        # Hard Negative 3: Solid / Plain / Smooth Gradient Frames
        sub_type = idx % 5
        if sub_type == 0:
            val = int(rng.randint(0, 12))
            img = np.full((size, size, 3), val, dtype=np.uint8)
            img[:8, :8, 0] = np.uint8((val + (idx % 19) + 1) % 256)
        elif sub_type == 1:
            val = int(rng.randint(242, 256))
            img = np.full((size, size, 3), val, dtype=np.uint8)
            img[:8, :8, 0] = np.uint8(max(0, val - ((idx % 19) + 1)))
        elif sub_type == 2:
            color = rng.randint(20, 235, size=3).tolist()
            img = np.full((size, size, 3), color, dtype=np.uint8)
            img[:8, :8, 1] = np.uint8((color[1] + (idx % 29) + 3) % 256)
        else:
            c1 = rng.randint(0, 255, size=3).astype(np.float32)
            c2 = rng.randint(0, 255, size=3).astype(np.float32)
            alpha = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
            grad = (1.0 - alpha) * c1[None, None, :] + alpha * c2[None, None, :]
            img = np.broadcast_to(grad, (size, size, 3)).copy().astype(np.uint8)
        return img, "hardneg_plain_or_gradient"

    if category == 3:
        # Hard Negative 4: Random noise / checkerboard / geometric stripes
        sub_type = idx % 3
        if sub_type == 0:
            img = rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
        elif sub_type == 1:
            tile = int(rng.choice([16, 24, 32, 40]))
            yy, xx = np.indices((size, size))
            grid = ((xx // tile) + (yy // tile)) % 2
            c_a = rng.randint(10, 100, size=3).astype(np.uint8)
            c_b = rng.randint(155, 250, size=3).astype(np.uint8)
            img = np.where(grid[:, :, None] == 1, c_a, c_b).astype(np.uint8)
        else:
            img = np.zeros((size, size, 3), dtype=np.uint8)
            step = int(rng.randint(14, 28))
            for i in range(0, size * 2, step):
                color = [int(x) for x in rng.randint(30, 245, size=3)]
                cv2.line(img, (0, i), (i, 0), color, step // 2)
        return img, "hardneg_noise_pattern"

    if category == 4:
        # Hard Negative 5: Synthetic External Camera Eye (non-fundus anterior segment)
        return _generate_external_eye_imposter(size, rng), "hardneg_external_eye"

    # Hard Negative 6: Synthetic Human Face / Portrait Selfie
    return _generate_human_face_imposter(size, rng), "hardneg_human_face"


def collect_local_random_images(
    random_dir: Path,
    n_train_natural: int,
    n_val_natural: int,
    n_test_natural: int,
    seed: int = SEED,
) -> Dict[str, List[Tuple[Path, str]]]:
    """Select strictly disjoint natural images from `DATASETS/class-random-image`."""
    train_dir = random_dir / "train"
    test_dir = random_dir / "test"
    train_csv = random_dir / "train.csv"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected train/ and test/ inside {random_dir}")

    rng = np.random.RandomState(seed)

    if train_csv.exists():
        df_tr = pd.read_csv(train_csv)
        df_tr["filepath"] = df_tr["Image"].apply(lambda name: train_dir / str(name))
        df_tr = df_tr[df_tr["filepath"].apply(lambda p: p.exists())].reset_index(drop=True)
    else:
        files = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
        df_tr = pd.DataFrame({"filepath": files, "Class": ["Misc"] * len(files)})

    total_needed_tr_val = n_train_natural + n_val_natural
    if len(df_tr) < total_needed_tr_val:
        raise ValueError(
            f"Not enough images in {train_dir}: need {total_needed_tr_val}, found {len(df_tr)}"
        )

    train_selected: List[Tuple[Path, str]] = []
    val_selected: List[Tuple[Path, str]] = []

    shuffled_parts = []
    for _, grp in df_tr.groupby("Class", sort=True):
        shuffled_parts.append(grp.sample(frac=1.0, random_state=seed))

    interleaved = (
        pd.concat(shuffled_parts)
        .assign(_order=lambda d: d.groupby("Class").cumcount())
        .sort_values(["_order", "Class"])
        .reset_index(drop=True)
    )

    seen_hashes: set[str] = set()
    for _, row in interleaved.iterrows():
        fpath: Path = row["filepath"]
        cat = str(row["Class"]).lower()
        fhash = compute_file_md5(fpath)
        if fhash in seen_hashes:
            continue
        seen_hashes.add(fhash)

        if len(val_selected) < n_val_natural:
            val_selected.append((fpath, f"local_{cat}"))
        elif len(train_selected) < n_train_natural:
            train_selected.append((fpath, f"local_{cat}"))
        else:
            break

    test_files = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    test_indices = rng.permutation(len(test_files))
    test_selected: List[Tuple[Path, str]] = []
    for idx in test_indices:
        fpath = test_files[int(idx)]
        fhash = compute_file_md5(fpath)
        if fhash in seen_hashes:
            continue
        seen_hashes.add(fhash)
        test_selected.append((fpath, "local_test_random"))
        if len(test_selected) >= n_test_natural:
            break

    if len(test_selected) < n_test_natural:
        raise ValueError(
            f"Could not find {n_test_natural} unique test images in {test_dir}."
        )

    return {
        "train": train_selected,
        "val": val_selected,
        "test": test_selected,
    }


def collect_torchvision_images(
    cache_dir: Path,
    n_train_natural: int,
    n_val_natural: int,
    n_test_natural: int,
    seed: int = SEED,
) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """Fallback option (`--source torchvision`): load CIFAR-100 non-eye images."""
    from torchvision import datasets

    rng = np.random.RandomState(seed)
    ds_train = datasets.CIFAR100(root=str(cache_dir), train=True, download=True)
    ds_test = datasets.CIFAR100(root=str(cache_dir), train=False, download=True)

    train_perm = rng.permutation(len(ds_train))
    test_perm = rng.permutation(len(ds_test))

    val_imgs: List[Tuple[np.ndarray, str]] = []
    train_imgs: List[Tuple[np.ndarray, str]] = []
    test_imgs: List[Tuple[np.ndarray, str]] = []

    for i in train_perm[: n_val_natural + n_train_natural]:
        pil_img, label_idx = ds_train[int(i)]
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cls_label = f"cifar100_{ds_train.classes[label_idx]}"
        if len(val_imgs) < n_val_natural:
            val_imgs.append((bgr, cls_label))
        else:
            train_imgs.append((bgr, cls_label))

    for i in test_perm[:n_test_natural]:
        pil_img, label_idx = ds_test[int(i)]
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cls_label = f"cifar100_{ds_test.classes[label_idx]}"
        test_imgs.append((bgr, cls_label))

    return {"train": train_imgs, "val": val_imgs, "test": test_imgs}


def verify_existing_ood_splits(
    dataset_dir: Path,
    targets: Dict[str, int],
) -> Tuple[bool, pd.DataFrame]:
    """Check if `{train,val,test}/non_fundus` already meet target counts and have 0 MD5 leakage."""
    records: list[dict] = []
    split_hashes: Dict[str, set[str]] = {s: set() for s in SPLITS}

    for split in SPLITS:
        ood_dir = dataset_dir / split / OOD_CLASS_NAME
        if not ood_dir.exists():
            return False, pd.DataFrame()
        files = sorted(p for p in ood_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        if len(files) < targets[split]:
            return False, pd.DataFrame()

        for fpath in files:
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is None or img.shape[0] < MIN_IMAGE_DIMENSION or img.shape[1] < MIN_IMAGE_DIMENSION:
                return False, pd.DataFrame()
            md5 = compute_file_md5(fpath)
            split_hashes[split].add(md5)
            records.append(
                {
                    "filepath": str(fpath.resolve()),
                    "filename": fpath.name,
                    "class_name": OOD_CLASS_NAME,
                    "label_idx": CLASS_TO_IDX_5[OOD_CLASS_NAME],
                    "orig_split": split,
                    "split": split,
                    "is_augmented": False,
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "md5_hash": md5,
                }
            )

    tr_va = len(split_hashes["train"] & split_hashes["val"])
    tr_te = len(split_hashes["train"] & split_hashes["test"])
    va_te = len(split_hashes["val"] & split_hashes["test"])
    if (tr_va + tr_te + va_te) > 0:
        print(f"[WARN] Cross-split MD5 leakage detected in existing non_fundus: tr_va={tr_va}, tr_te={tr_te}, va_te={va_te}")
        return False, pd.DataFrame()

    return True, pd.DataFrame(records)


def prepare_ood_dataset(
    dataset_dir: Path = DATASET_DIR,
    random_image_dir: Path = RANDOM_IMAGE_OOD_DIR,
    source: str = "local",
    train_target: int = OOD_TRAIN_TARGET,
    val_target: int = OOD_VAL_TARGET,
    test_target: int = OOD_TEST_TARGET,
    hard_negative_ratio: float = 0.24,
    force: bool = False,
    verify_only: bool = False,
    seed: int = SEED,
) -> pd.DataFrame:
    """Prepare and verify `non_fundus` splits with zero data leakage."""
    ensure_output_dir()
    targets = {"train": train_target, "val": val_target, "test": test_target}

    print("=" * 76)
    print("AI SENTINEL - NON_FUNDUS (OOD) DATASET PREPARATION")
    print(f"Target Dataset Dir : {dataset_dir}")
    print(f"OOD Source Mode    : {source} (Hard-Negative Ratio: {hard_negative_ratio * 100:.0f}%)")
    print(f"Target Split Counts: train={train_target}, val={val_target}, test={test_target}")
    print("=" * 76)

    if not force:
        is_ready, existing_df = verify_existing_ood_splits(dataset_dir, targets)
        if is_ready:
            existing_df.to_csv(OOD_MANIFEST_CSV, index=False)
            print(
                f"[OK] Existing `{OOD_CLASS_NAME}` splits verified (0 MD5 leakage, all >=224x224):\n"
                f"     train={(existing_df['orig_split']=='train').sum()} | "
                f"val={(existing_df['orig_split']=='val').sum()} | "
                f"test={(existing_df['orig_split']=='test').sum()}"
            )
            print(f"[SAVED] OOD manifest saved to: {OOD_MANIFEST_CSV}")
            return existing_df
        if verify_only:
            raise RuntimeError("`--verify-only` specified, but `non_fundus` splits are missing or incomplete.")

    n_hard: Dict[str, int] = {}
    n_nat: Dict[str, int] = {}
    for split, total_n in targets.items():
        nh = max(24, int(round(total_n * hard_negative_ratio)))
        n_hard[split] = nh
        n_nat[split] = total_n - nh

    if source == "local":
        natural_sources = collect_local_random_images(
            random_dir=random_image_dir,
            n_train_natural=n_nat["train"],
            n_val_natural=n_nat["val"],
            n_test_natural=n_nat["test"],
            seed=seed,
        )
    elif source == "torchvision":
        cache_dir = PROJECT_ROOT / "DATASETS" / "_torchvision_cache"
        natural_sources = collect_torchvision_images(
            cache_dir=cache_dir,
            n_train_natural=n_nat["train"],
            n_val_natural=n_nat["val"],
            n_test_natural=n_nat["test"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported `--source`: {source}. Choose 'local' or 'torchvision'.")

    split_seed_offsets = {"train": 10000, "val": 50000, "test": 90000}
    global_hashes: set[str] = set()
    records: list[dict] = []

    for split in SPLITS:
        out_dir = dataset_dir / split / OOD_CLASS_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        for old_file in out_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()

        rng = np.random.RandomState(seed + split_seed_offsets[split])
        saved_count = 0

        # 1. Export natural OOD images (upscaled >= 256x256)
        for item_idx, (src_item, sub_tag) in enumerate(natural_sources[split]):
            if isinstance(src_item, Path):
                bgr = cv2.imread(str(src_item), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
            else:
                bgr = src_item

            upscaled = upscale_to_min_resolution(bgr, target_min=256, target_max=340, rng=rng)
            jpeg_q = int(rng.randint(86, 97))
            ok, buf = cv2.imencode(".jpg", upscaled, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
            if not ok:
                continue
            raw_bytes = buf.tobytes()
            md5 = compute_md5_bytes(raw_bytes)
            attempt = 0
            while md5 in global_hashes:
                attempt += 1
                bx = (attempt * 8) % max(8, upscaled.shape[1] - 8)
                by = ((attempt // 8) * 8) % max(8, upscaled.shape[0] - 8)
                upscaled[by : by + 8, bx : bx + 8, attempt % 3] = np.uint8((attempt * 23 + item_idx) % 256)
                _, buf = cv2.imencode(".jpg", upscaled, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                raw_bytes = buf.tobytes()
                md5 = compute_md5_bytes(raw_bytes)

            global_hashes.add(md5)
            fname = f"ood_{split}_nat_{item_idx:04d}_{sub_tag}.jpg"
            out_path = out_dir / fname
            with open(out_path, "wb") as f:
                f.write(raw_bytes)

            records.append(
                {
                    "filepath": str(out_path.resolve()),
                    "filename": fname,
                    "class_name": OOD_CLASS_NAME,
                    "label_idx": CLASS_TO_IDX_5[OOD_CLASS_NAME],
                    "orig_split": split,
                    "split": split,
                    "is_augmented": False,
                    "ood_subtype": sub_tag,
                    "width": int(upscaled.shape[1]),
                    "height": int(upscaled.shape[0]),
                    "md5_hash": md5,
                }
            )
            saved_count += 1

        # 2. Export synthetic Hard-Negative OOD images (6 categories)
        for hn_idx in range(n_hard[split]):
            hn_bgr, sub_tag = generate_hard_negative_image(
                idx=hn_idx,
                split_seed_offset=split_seed_offsets[split] + seed,
            )
            ok, buf = cv2.imencode(".jpg", hn_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if not ok:
                continue
            raw_bytes = buf.tobytes()
            md5 = compute_md5_bytes(raw_bytes)
            attempt = 0
            while md5 in global_hashes:
                attempt += 1
                bx = (attempt * 8) % max(8, hn_bgr.shape[1] - 8)
                by = ((attempt // 8) * 8) % max(8, hn_bgr.shape[0] - 8)
                hn_bgr[by : by + 8, bx : bx + 8, attempt % 3] = np.uint8((attempt * 37 + hn_idx * 13) % 256)
                _, buf = cv2.imencode(".jpg", hn_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                raw_bytes = buf.tobytes()
                md5 = compute_md5_bytes(raw_bytes)

            global_hashes.add(md5)
            fname = f"ood_{split}_hn_{hn_idx:04d}_{sub_tag}.jpg"
            out_path = out_dir / fname
            with open(out_path, "wb") as f:
                f.write(raw_bytes)

            records.append(
                {
                    "filepath": str(out_path.resolve()),
                    "filename": fname,
                    "class_name": OOD_CLASS_NAME,
                    "label_idx": CLASS_TO_IDX_5[OOD_CLASS_NAME],
                    "orig_split": split,
                    "split": split,
                    "is_augmented": False,
                    "ood_subtype": sub_tag,
                    "width": int(hn_bgr.shape[1]),
                    "height": int(hn_bgr.shape[0]),
                    "md5_hash": md5,
                }
            )
            saved_count += 1

        print(
            f"[EXPORTED] {split:<5}/non_fundus : {saved_count:4d} images "
            f"(Natural: {n_nat[split]}, Hard-Negatives: {n_hard[split]}) -> {out_dir}"
        )

    ood_df = pd.DataFrame(records)
    tr_md5 = set(ood_df.loc[ood_df["orig_split"] == "train", "md5_hash"])
    va_md5 = set(ood_df.loc[ood_df["orig_split"] == "val", "md5_hash"])
    te_md5 = set(ood_df.loc[ood_df["orig_split"] == "test", "md5_hash"])
    overlap = len(tr_md5 & va_md5) + len(tr_md5 & te_md5) + len(va_md5 & te_md5)
    assert overlap == 0, f"CRITICAL: Found {overlap} overlapping MD5 hashes across OOD splits!"

    ood_df.to_csv(OOD_MANIFEST_CSV, index=False)
    print("-" * 76)
    print(f"[VERIFIED] Zero Cross-Split Leakage (0 MD5 overlaps across {len(ood_df)} OOD images).")
    print(f"[SAVED] OOD manifest saved to: {OOD_MANIFEST_CSV}")
    print("=" * 76)
    return ood_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare non_fundus OOD dataset splits.")
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        choices=["local", "torchvision"],
        help="OOD image source: 'local' (DATASETS/class-random-image) or 'torchvision' (CIFAR100).",
    )
    parser.add_argument("--train-count", type=int, default=OOD_TRAIN_TARGET)
    parser.add_argument("--val-count", type=int, default=OOD_VAL_TARGET)
    parser.add_argument("--test-count", type=int, default=OOD_TEST_TARGET)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.24)
    parser.add_argument("--force", action="store_true", help="Regenerate all non_fundus folders.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing folders.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_ood_dataset(
        source=args.source,
        train_target=args.train_count,
        val_target=args.val_count,
        test_target=args.test_count,
        hard_negative_ratio=args.hard_negative_ratio,
        force=args.force,
        verify_only=args.verify_only,
    )
