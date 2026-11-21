"""Comprehensive In-Memory QA & AI Robustness Benchmark Suite for IrisVision AI (5-Class + Gatekeeper).

Executes 3 core V&V / Software Testing & Network Efficiency experiments across ALL unseen
ocular fundus test images of `dataset2_balanced(Updated)` on BOTH 5-class models
(ResNet-50 & MobileNetV2) paired with Layer 1 Binary Gatekeeper:

1. Compression Robustness (`run_compression_experiment`):
   - 10 JPEG quality levels: Q = 100, 95, 85, 75, 60, 50, 40, 30, 20, 10
   - 3 WebP quality levels : Q = 80, 50, 20
   - 2 WhatsApp simulations: max-side 1280px @ Q=65, max-side 800px @ Q=65
   - Computes 4-class renormalized Accuracy, Macro/Weighted F1, per-class Recall (NORMAL,
     KATARAK, DR, GLAUKOMA), Mean Confidence, Delta Confidence, % below 90% & 70% threshold,
     Prediction Flip Rate (%), Layer-1 Gatekeeper False Rejection Rate (%), Layer-2 OOD
     False Rejection Rate (%), Combined 2-Layer False Rejection Rate (%), Payload Size (KB),
     Bandwidth Saving (%), PSNR (dB), and Latency (ms).

2. Format & Channel Compatibility (`run_format_experiment`):
   - Lossless PNG vs Lossy JPEG (Q=95), 4-channel RGBA PNG, 1-channel Grayscale PNG,
     WebP (Q=80), and uncompressed BMP, verified via `FundusImage`.

3. Resolution & Aspect Ratio Stress-Test (`run_resolution_experiment`):
   - Resolutions: 64x64, 112x112, 160x160, 224x224 (Native SRS boundary),
     512x512, 1024x1024, 2048x2048.
   - Aspect ratios: 1:1 (512x512), 4:3 (640x480), 16:9 (640x360).

All encoding/decoding runs strictly in memory (`cv2.imencode` & `io.BytesIO`).
Outputs CSV reports and 4 publication-grade PNG charts to `outputs/robustness/`.
"""

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_TO_IDX,
    CLASSES,
    CLASSES_5,
    CONFIDENCE_THRESHOLD,
    DATASET_DIR,
    GATEKEEPER_BEST_PATH,
    GATEKEEPER_CONFIG_JSON,
    GATEKEEPER_EYE_THRESHOLD,
    IMAGE_SIZE,
    L2_NON_FUNDUS_THRESHOLD,
    MIN_IMAGE_DIMENSION,
    MOBILENETV2_5CLASS_CONFIG_JSON,
    MOBILENETV2_5CLASS_PATH,
    OOD_CLASS_NAME,
    RESNET50_5CLASS_CONFIG_JSON,
    RESNET50_5CLASS_PATH,
    ROBUSTNESS_OUTPUT_DIR,
    USE_AMP,
    get_device,
)
from src.dataset import get_5class_split_dataframes, get_default_transforms
from src.evaluate import GradCAM, load_trained_model
from src.image_validator import FundusImage
from src.preprocess import preprocess_image

ROBUSTNESS_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

MODELS_TO_EVALUATE = [
    (
        "ResNet-50",
        RESNET50_5CLASS_PATH,
        RESNET50_5CLASS_CONFIG_JSON,
    ),
    (
        "MobileNetV2",
        MOBILENETV2_5CLASS_PATH,
        MOBILENETV2_5CLASS_CONFIG_JSON,
    ),
]


# ==============================================================================
# Core In-Memory Encoding, PSNR, & Batched GPU Inference Helpers
# ==============================================================================
def compute_psnr(orig_bgr: np.ndarray, comp_bgr: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between original and reconstructed image."""
    if orig_bgr.shape != comp_bgr.shape:
        comp_bgr = cv2.resize(
            comp_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_CUBIC
        )
    mse = float(np.mean((orig_bgr.astype(np.float32) - comp_bgr.astype(np.float32)) ** 2))
    if mse <= 1e-10:
        return 60.0
    return round(10.0 * math.log10((255.0 ** 2) / mse), 2)


def load_test_set_in_memory() -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Load unseen ocular fundus test images from `dataset2_balanced(Updated)` into memory."""
    _, _, test_df_5 = get_5class_split_dataframes(
        dataset_dir=DATASET_DIR,
        balance_train=False,
    )
    eye_test_df = (
        test_df_5[test_df_5["class_name"] != OOD_CLASS_NAME]
        .sort_values("filepath")
        .reset_index(drop=True)
    )
    filepaths = eye_test_df["filepath"].tolist()
    labels = eye_test_df["class_name"].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)

    def _read_one(p: str) -> np.ndarray:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read test image: {p}")
        return img

    with ThreadPoolExecutor(max_workers=8) as ex:
        raw_bgr_list = list(ex.map(_read_one, filepaths))

    print(
        f"[LOAD] Loaded {len(raw_bgr_list)} unseen ocular fundus test images from "
        f"`{DATASET_DIR.name}` into memory."
    )
    return raw_bgr_list, labels, filepaths


def transform_and_preprocess_batch(
    raw_bgr_list: List[np.ndarray],
    encode_fn: Callable[[np.ndarray], Tuple[bytes, str]],
    min_dim_check: int = 1,
) -> Tuple[torch.Tensor, float, float, float]:
    """Encode -> validate via FundusImage -> decode -> preprocess -> tensor batch in parallel."""
    eval_transform = get_default_transforms(is_train=False)

    def _process_one(orig_bgr: np.ndarray) -> Tuple[torch.Tensor, float, float]:
        raw_bytes, ext_name = encode_fn(orig_bgr)
        size_kb = len(raw_bytes) / 1024.0

        fimg = FundusImage(raw_bytes, filename=f"mem{ext_name}")
        fimg.validateFormat(
            allowed_formats=ROBUSTNESS_ALLOWED_FORMATS,
            min_width=min_dim_check,
            min_height=min_dim_check,
            check_extension=True,
        )
        decoded_bgr = fimg.to_bgr_array()
        psnr_val = compute_psnr(orig_bgr, decoded_bgr)
        prep_rgb = preprocess_image(decoded_bgr, class_name=None, target_size=IMAGE_SIZE, return_rgb=True)
        tensor_img = eval_transform(Image.fromarray(prep_rgb))
        return tensor_img, size_kb, psnr_val

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_process_one, raw_bgr_list))
    prep_ms_per_img = (time.perf_counter() - t0) * 1000.0 / max(1, len(raw_bgr_list))

    tensors = torch.stack([r[0] for r in results], dim=0)
    mean_kb = float(np.mean([r[1] for r in results]))
    mean_psnr = float(np.mean([r[2] for r in results]))
    return tensors, mean_kb, mean_psnr, prep_ms_per_img


@torch.no_grad()
def run_gatekeeper_batch(
    gk_model: Optional[nn.Module],
    tensors: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, float]:
    """Run Layer-1 Gatekeeper over batch; return (is_rejected_bool_array, gk_ms_per_img)."""
    n = tensors.size(0)
    if gk_model is None:
        return np.zeros(n, dtype=bool), 0.0

    gk_model.eval()
    use_amp = (device.type == "cuda") and USE_AMP
    probs_eye_list: List[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for start in range(0, n, batch_size):
        batch = tensors[start : start + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = gk_model(batch)
            probs = F.softmax(logits, dim=1)
        probs_eye_list.append(probs[:, 1].cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    gk_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n)

    p_valid_eye = np.concatenate(probs_eye_list)
    return (p_valid_eye < GATEKEEPER_EYE_THRESHOLD), gk_ms


@torch.no_grad()
def run_batched_inference(
    model: nn.Module,
    model_classes: List[str],
    tensors: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run batched GPU AMP inference supporting both 5-class and 4-class models.

    Returns:
        (renorm_4cls_preds [0..3], renorm_4cls_confs, l2_ood_rejected_bool, infer_ms_per_img)
    """
    model.eval()
    n = tensors.size(0)
    preds_list: List[np.ndarray] = []
    confs_list: List[np.ndarray] = []
    l2_rej_list: List[np.ndarray] = []
    use_amp = (device.type == "cuda") and USE_AMP

    has_ood = OOD_CLASS_NAME in model_classes
    ood_idx = model_classes.index(OOD_CLASS_NAME) if has_ood else -1
    med_indices = [model_classes.index(c) for c in CLASSES]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for start in range(0, n, batch_size):
        batch = tensors[start : start + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch)
            raw_probs = F.softmax(logits, dim=1)

        raw_np = raw_probs.cpu().numpy()
        if has_ood:
            raw_pred = np.argmax(raw_np, axis=1)
            l2_rej = (raw_pred == ood_idx) | (raw_np[:, ood_idx] > L2_NON_FUNDUS_THRESHOLD)
        else:
            l2_rej = np.zeros(raw_np.shape[0], dtype=bool)

        med_raw = raw_np[:, med_indices]
        med_sum = np.maximum(1e-8, med_raw.sum(axis=1, keepdims=True))
        med_renorm = med_raw / med_sum
        pred = np.argmax(med_renorm, axis=1)
        conf = np.max(med_renorm, axis=1)

        preds_list.append(pred)
        confs_list.append(conf)
        l2_rej_list.append(l2_rej)

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n)

    return (
        np.concatenate(preds_list),
        np.concatenate(confs_list),
        np.concatenate(l2_rej_list),
        infer_ms,
    )


def compute_metrics_row(
    model_name: str,
    y_true: np.ndarray,
    preds: np.ndarray,
    confs: np.ndarray,
    gk_rejected: np.ndarray,
    l2_rejected: np.ndarray,
    base_preds: np.ndarray,
    base_mean_conf: float,
    base_kb: float,
    mean_kb: float,
    mean_psnr: float,
    latency_ms: float,
    extra_meta: dict,
) -> dict:
    """Compute classification, per-class sensitivity, stability, OOD false-reject, and network metrics."""
    acc = float(accuracy_score(y_true, preds)) * 100.0
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0)) * 100.0
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0)) * 100.0

    recalls = recall_score(
        y_true, preds, labels=list(range(len(CLASSES))), average=None, zero_division=0
    ) * 100.0
    rec_map = {cls_name: float(r) for cls_name, r in zip(CLASSES, recalls)}

    mean_conf = float(np.mean(confs)) * 100.0
    delta_conf = round(mean_conf - base_mean_conf, 2)
    pct_below_90 = float(np.mean(confs < CONFIDENCE_THRESHOLD)) * 100.0
    pct_below_70 = float(np.mean(confs < 0.70)) * 100.0
    flip_rate = float(np.mean(preds != base_preds)) * 100.0
    bw_saving = (1.0 - (mean_kb / max(base_kb, 1e-6))) * 100.0

    gk_fr_pct = float(np.mean(gk_rejected)) * 100.0
    l2_fr_pct = float(np.mean(l2_rejected)) * 100.0
    comb_fr_pct = float(np.mean(gk_rejected | l2_rejected)) * 100.0

    return {
        "model": model_name,
        **extra_meta,
        "n_samples": int(len(y_true)),
        "accuracy_pct": round(acc, 2),
        "macro_f1_pct": round(macro_f1, 2),
        "weighted_f1_pct": round(weighted_f1, 2),
        "recall_normal_pct": round(rec_map["normal"], 2),
        "recall_katarak_pct": round(rec_map["cataract"], 2),
        "recall_dr_pct": round(rec_map["diabetic_retinopathy"], 2),
        "recall_glaukoma_pct": round(rec_map["glaucoma"], 2),
        "mean_confidence_pct": round(mean_conf, 2),
        "delta_confidence_pct": delta_conf,
        "pct_below_90_threshold": round(pct_below_90, 2),
        "pct_below_70_threshold": round(pct_below_70, 2),
        "prediction_flip_rate_pct": round(flip_rate, 2),
        "gk_false_reject_pct": round(gk_fr_pct, 2),
        "l2_ood_false_reject_pct": round(l2_fr_pct, 2),
        "combined_ood_false_reject_pct": round(comb_fr_pct, 2),
        "Ukuran_KB": round(mean_kb, 2),
        "bandwidth_saving_pct": round(bw_saving, 2),
        "psnr_db": round(mean_psnr, 2),
        "latencyMs": round(latency_ms, 2),
    }


# ==============================================================================
# Experiment #1: Compression Robustness (10 JPEG + 3 WebP + 2 WhatsApp)
# ==============================================================================
def _whatsapp_encode(img_bgr: np.ndarray, max_side: int, quality: int = 65) -> Tuple[bytes, str]:
    h, w = img_bgr.shape[:2]
    scale = max_side / float(max(h, w))
    new_w = max(224, int(round(w * scale)))
    new_h = max(224, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode WhatsApp simulation image.")
    return buf.tobytes(), ".jpg"


def run_compression_experiment(
    models_dict: Dict[str, Tuple[nn.Module, List[str]]],
    gk_model: Optional[nn.Module],
    raw_bgr_list: List[np.ndarray],
    y_true: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Run in-memory compression experiments across 15 conditions on both 5-class models + Gatekeeper."""
    n_samples = len(raw_bgr_list)
    print("\n" + "=" * 84)
    print(f"STAGE 1: IN-MEMORY COMPRESSION ROBUSTNESS BENCHMARK (n={n_samples}, 5-Class + Gatekeeper)")
    print("=" * 84)

    conditions: List[Tuple[str, str, int, Callable[[np.ndarray], Tuple[bytes, str]]]] = []
    for q in [100, 95, 85, 75, 60, 50, 40, 30, 20, 10]:
        conditions.append(
            (
                f"JPEG_Q{q}",
                "JPEG",
                q,
                lambda img, q_val=q: (
                    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q_val])[1].tobytes(),
                    ".jpg",
                ),
            )
        )
    for wq in [80, 50, 20]:
        conditions.append(
            (
                f"WebP_Q{wq}",
                "WEBP",
                wq,
                lambda img, q_val=wq: (
                    cv2.imencode(".webp", img, [int(cv2.IMWRITE_WEBP_QUALITY), q_val])[1].tobytes(),
                    ".webp",
                ),
            )
        )
    conditions.append(("WhatsApp_1280_Q65", "WhatsApp", 65, lambda img: _whatsapp_encode(img, 1280, 65)))
    conditions.append(("WhatsApp_800_Q65", "WhatsApp", 65, lambda img: _whatsapp_encode(img, 800, 65)))

    rows: List[dict] = []
    baseline_cache: Dict[str, Tuple[np.ndarray, float, float]] = {}

    for cond_label, codec, quality, enc_fn in conditions:
        tensors, mean_kb, mean_psnr, prep_ms = transform_and_preprocess_batch(raw_bgr_list, enc_fn)
        gk_rej, gk_ms = run_gatekeeper_batch(gk_model, tensors, device)

        for model_name, (model, m_classes) in models_dict.items():
            preds, confs, l2_rej, infer_ms = run_batched_inference(model, m_classes, tensors, device)
            total_lat = prep_ms + gk_ms + infer_ms

            if cond_label == "JPEG_Q100":
                baseline_cache[model_name] = (
                    preds.copy(),
                    float(np.mean(confs)) * 100.0,
                    mean_kb,
                )

            base_preds, base_conf, base_kb = baseline_cache[model_name]
            row = compute_metrics_row(
                model_name=model_name,
                y_true=y_true,
                preds=preds,
                confs=confs,
                gk_rejected=gk_rej,
                l2_rejected=l2_rej,
                base_preds=base_preds,
                base_mean_conf=base_conf,
                base_kb=base_kb,
                mean_kb=mean_kb,
                mean_psnr=mean_psnr,
                latency_ms=total_lat,
                extra_meta={
                    "condition": cond_label,
                    "codec": codec,
                    "quality": quality,
                },
            )
            rows.append(row)
            print(
                f"  [{model_name:<11} | {cond_label:<17}] "
                f"Acc={row['accuracy_pct']:6.2f}% | MacroF1={row['macro_f1_pct']:6.2f}% | "
                f"OOD_FR={row['combined_ood_false_reject_pct']:5.2f}% | "
                f"Flip={row['prediction_flip_rate_pct']:5.2f}% | KB={row['Ukuran_KB']:6.1f}"
            )

    df_comp = pd.DataFrame(rows)
    csv_path = ROBUSTNESS_OUTPUT_DIR / "compression_test_results.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"[SAVED] Compression CSV saved to: {csv_path}")
    return df_comp


# ==============================================================================
# Experiment #2: Format & Channel Compatibility
# ==============================================================================
def run_format_experiment(
    models_dict: Dict[str, Tuple[nn.Module, List[str]]],
    gk_model: Optional[nn.Module],
    raw_bgr_list: List[np.ndarray],
    y_true: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Compare PNG Lossless vs JPEG Q=95, 4-ch RGBA, 1-ch Grayscale, WebP, and BMP."""
    n_samples = len(raw_bgr_list)
    print("\n" + "=" * 84)
    print(f"STAGE 2: FORMAT & CHANNEL COMPATIBILITY BENCHMARK (n={n_samples}, 5-Class + Gatekeeper)")
    print("=" * 84)

    def _to_rgba_png(img: np.ndarray) -> Tuple[bytes, str]:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        bgra = np.concatenate([img, alpha], axis=2)
        return cv2.imencode(".png", bgra)[1].tobytes(), ".png"

    def _to_gray_png(img: np.ndarray) -> Tuple[bytes, str]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.imencode(".png", gray)[1].tobytes(), ".png"

    format_specs = [
        ("PNG_Lossless_3Ch", "PNG", 3, lambda img: (cv2.imencode(".png", img)[1].tobytes(), ".png")),
        ("JPEG_Q95_3Ch", "JPEG", 3, lambda img: (cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes(), ".jpg")),
        ("PNG_RGBA_4Ch", "PNG", 4, _to_rgba_png),
        ("PNG_Grayscale_1Ch", "PNG", 1, _to_gray_png),
        ("WebP_Q80_3Ch", "WEBP", 3, lambda img: (cv2.imencode(".webp", img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])[1].tobytes(), ".webp")),
        ("BMP_Uncompressed_3Ch", "BMP", 3, lambda img: (cv2.imencode(".bmp", img)[1].tobytes(), ".bmp")),
    ]

    rows: List[dict] = []
    base_cache: Dict[str, Tuple[np.ndarray, float, float]] = {}

    for fmt_label, container, ch_count, enc_fn in format_specs:
        sample_bytes, sample_ext = enc_fn(raw_bgr_list[0])
        sample_fimg = FundusImage(sample_bytes, filename=f"sample{sample_ext}")
        sample_hash = sample_fimg.computeHash()[:16]

        tensors, mean_kb, mean_psnr, prep_ms = transform_and_preprocess_batch(raw_bgr_list, enc_fn)
        gk_rej, gk_ms = run_gatekeeper_batch(gk_model, tensors, device)

        for model_name, (model, m_classes) in models_dict.items():
            preds, confs, l2_rej, infer_ms = run_batched_inference(model, m_classes, tensors, device)
            total_lat = prep_ms + gk_ms + infer_ms

            if fmt_label == "PNG_Lossless_3Ch":
                base_cache[model_name] = (
                    preds.copy(),
                    float(np.mean(confs)) * 100.0,
                    mean_kb,
                )

            base_preds, base_conf, base_kb = base_cache[model_name]
            row = compute_metrics_row(
                model_name=model_name,
                y_true=y_true,
                preds=preds,
                confs=confs,
                gk_rejected=gk_rej,
                l2_rejected=l2_rej,
                base_preds=base_preds,
                base_mean_conf=base_conf,
                base_kb=base_kb,
                mean_kb=mean_kb,
                mean_psnr=mean_psnr,
                latency_ms=total_lat,
                extra_meta={
                    "format_variant": fmt_label,
                    "container": container,
                    "channels": ch_count,
                    "sample_sha256_prefix": sample_hash,
                },
            )
            rows.append(row)
            print(
                f"  [{model_name:<11} | {fmt_label:<20}] "
                f"Acc={row['accuracy_pct']:6.2f}% | OOD_FR={row['combined_ood_false_reject_pct']:5.2f}% | "
                f"Flip={row['prediction_flip_rate_pct']:5.2f}% | KB={row['Ukuran_KB']:6.1f}"
            )

    df_fmt = pd.DataFrame(rows)
    csv_path = ROBUSTNESS_OUTPUT_DIR / "format_test_results.csv"
    df_fmt.to_csv(csv_path, index=False)
    print(f"[SAVED] Format CSV saved to: {csv_path}")
    return df_fmt


# ==============================================================================
# Experiment #3: Resolution & Aspect Ratio Stress-Test
# ==============================================================================
def run_resolution_experiment(
    models_dict: Dict[str, Tuple[nn.Module, List[str]]],
    gk_model: Optional[nn.Module],
    raw_bgr_list: List[np.ndarray],
    y_true: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Test resolutions 64x64..2048x2048 and aspect ratios 1:1, 4:3, 16:9 on both models."""
    n_samples = len(raw_bgr_list)
    print("\n" + "=" * 84)
    print(f"STAGE 3: RESOLUTION & ASPECT RATIO STRESS-TEST (n={n_samples}, 5-Class + Gatekeeper)")
    print("=" * 84)

    res_specs = [
        ("64x64", "Square_Resolution", 64, 64, "1:1"),
        ("112x112", "Square_Resolution", 112, 112, "1:1"),
        ("160x160", "Square_Resolution", 160, 160, "1:1"),
        ("224x224", "Square_Resolution", 224, 224, "1:1"),
        ("512x512", "Square_Resolution", 512, 512, "1:1"),
        ("1024x1024", "Square_Resolution", 1024, 1024, "1:1"),
        ("2048x2048", "Square_Resolution", 2048, 2048, "1:1"),
        ("Aspect_1_1_512x512", "Aspect_Ratio", 512, 512, "1:1"),
        ("Aspect_4_3_640x480", "Aspect_Ratio", 640, 480, "4:3"),
        ("Aspect_16_9_640x360", "Aspect_Ratio", 640, 360, "16:9"),
    ]

    native_fn = lambda img: (
        cv2.imencode(
            ".jpg",
            cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )[1].tobytes(),
        ".jpg",
    )
    nat_tensors, nat_kb, _, _ = transform_and_preprocess_batch(raw_bgr_list, native_fn, min_dim_check=1)
    base_cache: Dict[str, Tuple[np.ndarray, float, float]] = {}
    for m_name, (model, m_classes) in models_dict.items():
        b_preds, b_confs, _, _ = run_batched_inference(model, m_classes, nat_tensors, device)
        base_cache[m_name] = (b_preds.copy(), float(np.mean(b_confs)) * 100.0, nat_kb)

    rows: List[dict] = []
    for res_label, category, w_px, h_px, ratio_str in res_specs:
        def _res_encode(img: np.ndarray, tw: int = w_px, th: int = h_px) -> Tuple[bytes, str]:
            h0, w0 = img.shape[:2]
            interp = cv2.INTER_AREA if (tw < w0 or th < h0) else cv2.INTER_CUBIC
            resized = cv2.resize(img, (tw, th), interpolation=interp)
            return (
                cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes(),
                ".jpg",
            )

        tensors, mean_kb, mean_psnr, prep_ms = transform_and_preprocess_batch(
            raw_bgr_list, _res_encode, min_dim_check=1
        )
        gk_rej, gk_ms = run_gatekeeper_batch(gk_model, tensors, device)
        srs_accepted = bool(w_px >= MIN_IMAGE_DIMENSION and h_px >= MIN_IMAGE_DIMENSION)

        for model_name, (model, m_classes) in models_dict.items():
            preds, confs, l2_rej, infer_ms = run_batched_inference(model, m_classes, tensors, device)
            total_lat = prep_ms + gk_ms + infer_ms
            base_preds, base_conf, base_kb = base_cache[model_name]

            row = compute_metrics_row(
                model_name=model_name,
                y_true=y_true,
                preds=preds,
                confs=confs,
                gk_rejected=gk_rej,
                l2_rejected=l2_rej,
                base_preds=base_preds,
                base_mean_conf=base_conf,
                base_kb=base_kb,
                mean_kb=mean_kb,
                mean_psnr=mean_psnr,
                latency_ms=total_lat,
                extra_meta={
                    "resolution_label": res_label,
                    "category": category,
                    "lebarPx": w_px,
                    "tinggiPx": h_px,
                    "aspect_ratio": ratio_str,
                    "srs_accepted_min_224px": srs_accepted,
                },
            )
            rows.append(row)
            print(
                f"  [{model_name:<11} | {res_label:<19}] "
                f"Acc={row['accuracy_pct']:6.2f}% | MacroF1={row['macro_f1_pct']:6.2f}% | "
                f"OOD_FR={row['combined_ood_false_reject_pct']:5.2f}% | "
                f"Latency={row['latencyMs']:6.2f}ms | SRS_Valid={srs_accepted}"
            )

    df_res = pd.DataFrame(rows)
    csv_path = ROBUSTNESS_OUTPUT_DIR / "resolution_test_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"[SAVED] Resolution CSV saved to: {csv_path}")
    return df_res


# ==============================================================================
# Visualization Generators (Plots 1, 2, 3, and 4)
# ==============================================================================
def plot_1_compression_accuracy_and_confidence(df_comp: pd.DataFrame, n_samples: int) -> Dict[str, int]:
    """Generate 1_compression_accuracy_and_confidence.png and return min safe Q per model."""
    sns.set_theme(style="whitegrid", context="notebook")
    jpeg_df = df_comp[df_comp["codec"] == "JPEG"].sort_values("quality", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), constrained_layout=True)
    safe_q_map: Dict[str, int] = {}

    for ax, model_name in zip(axes, ["ResNet-50", "MobileNetV2"]):
        sub = jpeg_df[jpeg_df["model"] == model_name].sort_values("quality", ascending=False)
        q_vals = sub["quality"].to_numpy()
        base_acc = float(sub.loc[sub["quality"] == 100, "accuracy_pct"].iloc[0])

        safe_rows = sub[(base_acc - sub["accuracy_pct"]) < 1.5]
        min_safe_q = int(safe_rows["quality"].min()) if not safe_rows.empty else 100
        safe_q_map[model_name] = min_safe_q

        ax.plot(q_vals, sub["accuracy_pct"], marker="o", linewidth=2.4, color="#1D3557", label="Accuracy (%)")
        ax.plot(q_vals, sub["macro_f1_pct"], marker="s", linewidth=2.2, color="#2A9D8F", label="Macro F1 (%)")
        ax.plot(q_vals, sub["mean_confidence_pct"], marker="^", linewidth=2.0, linestyle="--", color="#457B9D", label="Mean Confidence (%)")

        ax2 = ax.twinx()
        ax2.plot(q_vals, sub["prediction_flip_rate_pct"], marker="D", linewidth=2.2, color="#E63946", label="Prediction Flip Rate (%)")
        ax2.plot(q_vals, sub["combined_ood_false_reject_pct"], marker="v", linewidth=1.8, linestyle=":", color="#F4A261", label="2-Layer OOD False Reject (%)")
        ax2.set_ylabel("Flip Rate / OOD False Reject (%)", color="#E63946", fontweight="bold")
        ax2.grid(False)
        ax2.set_ylim(-1, max(35.0, float(max(sub["prediction_flip_rate_pct"].max(), sub["combined_ood_false_reject_pct"].max())) + 5.0))

        ax.axvline(
            min_safe_q,
            color="#D62828",
            linestyle="-.",
            linewidth=2.0,
            label=f"Min Safe JPEG Q={min_safe_q} (ΔAcc < 1.5%)",
        )

        ax.invert_xaxis()
        ax.set_xticks([100, 95, 85, 75, 60, 50, 40, 30, 20, 10])
        ax.set_ylim(max(65.0, float(sub["macro_f1_pct"].min()) - 4.0), 100.5)
        ax.set_xlabel("JPEG Quality Factor (Q=100 → Q=10)", fontweight="bold")
        ax.set_ylabel("Accuracy / Macro F1 / Confidence (%)", fontweight="bold")
        ax.set_title(f"{model_name} (5-Class + Gatekeeper) — Compression Stability", fontsize=12.5, fontweight="bold")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.5, frameon=True)

    fig.suptitle(
        f"IrisVision AI (5-Class + Gatekeeper) — JPEG Compression vs Confidence, Flip Rate & OOD FR (n={n_samples})",
        fontsize=14.5,
        fontweight="bold",
    )
    out_path = ROBUSTNESS_OUTPUT_DIR / "1_compression_accuracy_and_confidence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot 1 saved to: {out_path}")
    return safe_q_map


def plot_2_per_class_compression_sensitivity(df_comp: pd.DataFrame) -> None:
    """Generate 2_per_class_compression_sensitivity.png for NORMAL, KATARAK, DR, GLAUKOMA."""
    sns.set_theme(style="whitegrid", context="notebook")
    jpeg_df = df_comp[df_comp["codec"] == "JPEG"].sort_values("quality", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.2), constrained_layout=True)
    class_curves = [
        ("recall_normal_pct", "NORMAL", "#2A9D8F", "o"),
        ("recall_katarak_pct", "KATARAK (Fundus)", "#E76F51", "s"),
        ("recall_dr_pct", "DR (Diabetic Retinopathy)", "#1D3557", "^"),
        ("recall_glaukoma_pct", "GLAUKOMA", "#9B5DE5", "D"),
    ]

    for ax, model_name in zip(axes, ["ResNet-50", "MobileNetV2"]):
        sub = jpeg_df[jpeg_df["model"] == model_name].sort_values("quality", ascending=False)
        q_vals = sub["quality"].to_numpy()

        for col, label, color, marker in class_curves:
            ax.plot(q_vals, sub[col], marker=marker, linewidth=2.3, color=color, label=f"Recall {label}")

        ax.invert_xaxis()
        ax.set_xticks([100, 95, 85, 75, 60, 50, 40, 30, 20, 10])
        min_rec = float(sub[["recall_normal_pct", "recall_katarak_pct", "recall_dr_pct", "recall_glaukoma_pct"]].min().min())
        ax.set_ylim(max(50.0, min_rec - 5.0), 101.5)
        ax.set_xlabel("JPEG Quality Factor (Q=100 → Q=10)", fontweight="bold")
        ax.set_ylabel("Per-Class Recall / Sensitivity (%)", fontweight="bold")
        ax.set_title(f"{model_name} (5-Class) — Per-Class Compression Sensitivity", fontsize=12.5, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9, frameon=True)

    fig.suptitle(
        "IrisVision AI (5-Class) — Per-Disease Sensitivity (Recall %) Across Lossy JPEG Compression Levels",
        fontsize=14.5,
        fontweight="bold",
    )
    out_path = ROBUSTNESS_OUTPUT_DIR / "2_per_class_compression_sensitivity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot 2 saved to: {out_path}")


def plot_3_compression_artifacts_and_gradcam(
    model: nn.Module,
    model_classes: List[str],
    raw_bgr_list: List[np.ndarray],
    y_true: np.ndarray,
    device: torch.device,
) -> None:
    """Generate 3_compression_artifacts_and_gradcam.png across Q=100, Q=75, Q=40, Q=15."""
    dr_idx = CLASSES.index("diabetic_retinopathy")
    cand_indices = np.where(y_true == dr_idx)[0]
    sample_bgr = raw_bgr_list[int(cand_indices[0])] if len(cand_indices) > 0 else raw_bgr_list[0]

    target_layer = model.get_target_layer_for_cam()
    cam_extractor = GradCAM(model, target_layer)
    eval_transform = get_default_transforms(is_train=False)

    qualities = [100, 75, 40, 15]
    fig, axes = plt.subplots(3, 4, figsize=(17, 12.5), constrained_layout=True)

    _, buf100 = cv2.imencode(".jpg", sample_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    bgr_100 = cv2.imdecode(buf100, cv2.IMREAD_COLOR)
    rgb_100 = cv2.cvtColor(cv2.resize(bgr_100, IMAGE_SIZE), cv2.COLOR_BGR2RGB)

    for col_i, q in enumerate(qualities):
        _, buf = cv2.imencode(".jpg", sample_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        comp_bytes = buf.tobytes()
        size_kb = len(comp_bytes) / 1024.0
        comp_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        psnr_val = compute_psnr(bgr_100, comp_bgr)

        comp_rgb_224 = cv2.cvtColor(cv2.resize(comp_bgr, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        diff_amp = np.clip(
            np.abs(rgb_100.astype(np.float32) - comp_rgb_224.astype(np.float32)) * 5.0,
            0,
            255,
        ).astype(np.uint8)

        prep_rgb = preprocess_image(comp_bgr, class_name=None, target_size=IMAGE_SIZE, return_rgb=True)
        inp = eval_transform(Image.fromarray(prep_rgb)).unsqueeze(0).to(device)
        cam_map, pred_idx, pred_conf = cam_extractor.generate(inp)
        pred_cls = model_classes[pred_idx]

        heat_bgr = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = np.clip(
            0.56 * prep_rgb.astype(np.float32) + 0.44 * heat_rgb.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        axes[0, col_i].imshow(comp_rgb_224)
        axes[0, col_i].set_title(
            f"JPEG Q={q} {'(Original)' if q == 100 else ''}\n"
            f"Size: {size_kb:.1f} KB | PSNR: {psnr_val:.1f} dB",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[0, col_i].axis("off")

        axes[1, col_i].imshow(diff_amp)
        mean_err = float(np.abs(rgb_100.astype(np.float32) - comp_rgb_224.astype(np.float32)).mean())
        axes[1, col_i].set_title(
            f"5× Amplified Error |I_orig - I_Q{q}|\nMean Pixel Error: {mean_err:.2f} / 255",
            fontsize=10,
            fontweight="bold",
        )
        axes[1, col_i].axis("off")

        axes[2, col_i].imshow(overlay_rgb)
        axes[2, col_i].set_title(
            f"Grad-CAM Overlay (Q={q})\nPred: {pred_cls} ({pred_conf * 100:.2f}%)",
            fontsize=10,
            fontweight="bold",
        )
        axes[2, col_i].axis("off")

    cam_extractor.close()
    fig.suptitle(
        "IrisVision AI (5-Class) — JPEG 8×8 Block Artifacts, Pixel Error Map (×5), & Grad-CAM Drift",
        fontsize=15,
        fontweight="bold",
    )
    out_path = ROBUSTNESS_OUTPUT_DIR / "3_compression_artifacts_and_gradcam.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot 3 saved to: {out_path}")


def plot_4_resolution_vs_accuracy_latency(df_res: pd.DataFrame) -> None:
    """Generate 4_resolution_vs_accuracy_latency.png for square resolutions and aspect ratios."""
    sns.set_theme(style="whitegrid", context="notebook")
    sq_df = df_res[df_res["category"] == "Square_Resolution"].sort_values("lebarPx")
    ar_df = df_res[df_res["category"] == "Aspect_Ratio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.2), constrained_layout=True)

    res_labels = ["64x64", "112x112", "160x160", "224x224", "512x512", "1024x1024", "2048x2048"]
    x_pos = np.arange(len(res_labels))

    for model_name, color, marker in [("ResNet-50", "#1D3557", "o"), ("MobileNetV2", "#2A9D8F", "s")]:
        sub = sq_df[sq_df["model"] == model_name].set_index("resolution_label").reindex(res_labels)
        ax1.plot(x_pos, sub["accuracy_pct"], marker=marker, linewidth=2.4, color=color, label=f"{model_name} Accuracy (%)")
        ax1.plot(x_pos, sub["macro_f1_pct"], marker=marker, linewidth=1.8, linestyle="--", color=color, alpha=0.75, label=f"{model_name} Macro F1 (%)")

    ax1_lat = ax1.twinx()
    for model_name, color in [("ResNet-50", "#E63946"), ("MobileNetV2", "#F4A261")]:
        sub = sq_df[sq_df["model"] == model_name].set_index("resolution_label").reindex(res_labels)
        ax1_lat.plot(x_pos, sub["latencyMs"], marker="^", linewidth=2.0, linestyle=":", color=color, label=f"{model_name} 2-Layer Latency (ms)")
    ax1_lat.set_ylabel("Processing + 2-Layer Inference Latency (ms/image)", color="#E63946", fontweight="bold")
    ax1_lat.grid(False)

    ax1.axvline(3, color="#D62828", linestyle="-.", linewidth=2.2, label="SRS Min Resolution Boundary (224×224)")
    ax1.axvspan(-0.3, 2.5, color="#F8D7DA", alpha=0.32, label="Rejected Zone (< 224×224)")

    ax1.set_xticks(x_pos, res_labels, rotation=20)
    ax1.set_xlabel("Input Image Resolution (px)", fontweight="bold")
    ax1.set_ylabel("Accuracy / Macro F1 (%)", fontweight="bold")
    ax1.set_title("Input Resolution vs. Classification Accuracy & 2-Layer Latency", fontsize=12.5, fontweight="bold")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_lat.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, frameon=True)

    sns.barplot(
        data=ar_df,
        x="aspect_ratio",
        y="accuracy_pct",
        hue="model",
        palette={"ResNet-50": "#1D3557", "MobileNetV2": "#2A9D8F"},
        ax=ax2,
    )
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.2f%%", fontsize=9, padding=3, fontweight="bold")
    ax2.set_ylim(75.0, 101.0)
    ax2.set_xlabel("Input Aspect Ratio (1:1 Square, 4:3 Standard, 16:9 Widescreen)", fontweight="bold")
    ax2.set_ylabel("4-Class Renormalized Accuracy (%)", fontweight="bold")
    ax2.set_title("Aspect Ratio Distortion vs. Test Accuracy", fontsize=12.5, fontweight="bold")
    ax2.legend(title="Model (5-Class)", loc="lower right")

    fig.suptitle(
        "IrisVision AI (5-Class + Gatekeeper) — Resolution Scaling & Aspect Ratio Robustness",
        fontsize=14.5,
        fontweight="bold",
    )
    out_path = ROBUSTNESS_OUTPUT_DIR / "4_resolution_vs_accuracy_latency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot 4 saved to: {out_path}")


# ==============================================================================
# Main Runner
# ==============================================================================
def run_all_robustness_tests() -> None:
    ROBUSTNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    gk_model: Optional[nn.Module] = None
    if GATEKEEPER_BEST_PATH.exists():
        gk_model, gk_cfg = load_trained_model(GATEKEEPER_BEST_PATH, GATEKEEPER_CONFIG_JSON, device=device)
        print(f"[MODEL] Loaded Layer-1 Gatekeeper ({gk_cfg.get('backbone', 'mobilenet_v2')}) on {device}")

    models_dict: Dict[str, Tuple[nn.Module, List[str]]] = {}
    for m_name, ckpt_path, cfg_path in MODELS_TO_EVALUATE:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Required 5-class model checkpoint not found: {ckpt_path}")
        model, cfg = load_trained_model(ckpt_path, cfg_path, device=device)
        m_classes = list(cfg.get("classes", CLASSES_5))
        models_dict[m_name] = (model, m_classes)
        print(f"[MODEL] Loaded {m_name} ({m_classes}) from {ckpt_path.name} on {device}")

    raw_bgr_list, y_true, _ = load_test_set_in_memory()

    df_comp = run_compression_experiment(models_dict, gk_model, raw_bgr_list, y_true, device)
    df_fmt = run_format_experiment(models_dict, gk_model, raw_bgr_list, y_true, device)
    df_res = run_resolution_experiment(models_dict, gk_model, raw_bgr_list, y_true, device)

    safe_q_map = plot_1_compression_accuracy_and_confidence(df_comp, len(raw_bgr_list))
    plot_2_per_class_compression_sensitivity(df_comp)
    rn_model, rn_classes = models_dict["ResNet-50"]
    plot_3_compression_artifacts_and_gradcam(rn_model, rn_classes, raw_bgr_list, y_true, device)
    plot_4_resolution_vs_accuracy_latency(df_res)

    print("\n" + "=" * 84)
    print("QA & AI ROBUSTNESS EXECUTIVE SUMMARY (5-CLASS + LAYER-1 GATEKEEPER)")
    print("=" * 84)
    for m_name in ["ResNet-50", "MobileNetV2"]:
        print(
            f"  * {m_name:<11} Minimum Acceptable JPEG Quality (ΔAcc < 1.5%): "
            f"Q = {safe_q_map.get(m_name, 100)}"
        )
    print(f"  * Format Compatibility rows saved : {len(df_fmt)}")
    print(f"  * Resolution Stress-Test rows saved: {len(df_res)}")
    print(f"  * All artifacts saved under       : {ROBUSTNESS_OUTPUT_DIR}")
    print("=" * 84)


if __name__ == "__main__":
    run_all_robustness_tests()
