"""2-way head-to-head 5-class model comparison (+ Layer 1 Gatekeeper) for Iris Vision.

Compares on the exact same 5-class unseen test split of `dataset2_balanced(Updated)`:
1. MobileNetV2 (5-Class + Gatekeeper) -> `models/mobilenetv2_5class/`
2. ResNet-50   (5-Class + Gatekeeper) -> `models/resnet50_5class/`

Evaluates:
- 5-Class overall accuracy, Macro Precision/Recall/F1, Weighted F1, and per-class F1
  across `['cataract', 'diabetic_retinopathy', 'glaucoma', 'non_fundus', 'normal']`
- Ocular 4-class renormalized accuracy (valid fundus subset)
- Layer-2 standalone OOD FAR / Rejection Rate & Combined 2-Layer (Gatekeeper + 5-Class)
  OOD FAR, OOD Rejection Rate, and Valid Fundus Pass Rate
- Model parameters (M), checkpoint size (MB), and GPU/CPU single-image latency (ms)
- Paired Exact McNemar statistical significance test (5-class & 4-class ocular subset)

Outputs:
- outputs/model_comparison_report.md
- outputs/model_comparison.png
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    CLASSES_5,
    DATASET_DIR,
    GATEKEEPER_BEST_PATH,
    GATEKEEPER_CONFIG_JSON,
    IMAGE_SIZE,
    MOBILENETV2_5CLASS_CONFIG_JSON,
    MOBILENETV2_5CLASS_PATH,
    OOD_CLASS_NAME,
    OUTPUTS_DIR,
    PIN_MEMORY,
    RESNET50_5CLASS_CONFIG_JSON,
    RESNET50_5CLASS_PATH,
    USE_AMP,
    ensure_output_dir,
    get_device,
)
from src.dataset import (
    EyeDiseaseDataset,
    get_5class_split_dataframes,
    get_default_transforms,
)
from src.evaluate import (
    compute_ood_and_combined_metrics,
    evaluate_test_set,
    load_trained_model,
)

COMPARISON_REPORT_MD: Path = OUTPUTS_DIR / "model_comparison_report.md"
COMPARISON_PNG: Path = OUTPUTS_DIR / "model_comparison.png"


@dataclass
class ModelSpec:
    label: str
    dataset_label: str
    checkpoint: Path
    config: Path


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        label="MobileNetV2 (5-Class)",
        dataset_label=DATASET_DIR.name,
        checkpoint=MOBILENETV2_5CLASS_PATH,
        config=MOBILENETV2_5CLASS_CONFIG_JSON,
    ),
    ModelSpec(
        label="ResNet-50 (5-Class)",
        dataset_label=DATASET_DIR.name,
        checkpoint=RESNET50_5CLASS_PATH,
        config=RESNET50_5CLASS_CONFIG_JSON,
    ),
]


@torch.no_grad()
def measure_latency_ms(model: torch.nn.Module, device: torch.device, n_iters: int = 100) -> float:
    """Average single-image (batch=1) forward latency in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, *IMAGE_SIZE, device=device)
    amp = (device.type == "cuda") and USE_AMP
    for _ in range(15):  # warm-up
        with torch.amp.autocast("cuda", enabled=amp):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.amp.autocast("cuda", enabled=amp):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iters


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on paired predictions over the same test set."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    p_value = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p_value}


def evaluate_spec(
    spec: ModelSpec,
    test_loader: DataLoader,
    test_files: list[str],
    device: torch.device,
    gk_latency_ms: float,
) -> Optional[dict]:
    if not spec.checkpoint.exists():
        print(f"[SKIP] {spec.label}: checkpoint not found at {spec.checkpoint}")
        return None

    print(f"\n[EVAL] {spec.label} ({spec.checkpoint.name}) ...")
    model, config = load_trained_model(spec.checkpoint, spec.config, device=device)
    active_classes = list(config.get("classes", CLASSES_5))

    y_true, y_pred, y_probs = evaluate_test_set(model, test_loader, device=device, use_amp=USE_AMP)
    labels = list(range(len(CLASSES_5)))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    ood_metrics = compute_ood_and_combined_metrics(
        y_true, y_pred, y_probs, active_classes, test_loader, device
    )

    # Renormalized 4-class predictions on valid ocular test images (for paired ocular McNemar)
    ood_idx = active_classes.index(OOD_CLASS_NAME)
    eye_mask = y_true != ood_idx
    med_indices = [i for i, c in enumerate(active_classes) if c != OOD_CLASS_NAME]
    med_pred_local = np.argmax(y_probs[:, med_indices], axis=1)
    med_pred_global = np.array([med_indices[i] for i in med_pred_local], dtype=np.int64)

    standalone_ms = measure_latency_ms(model, device)
    result = {
        "label": spec.label,
        "dataset": spec.dataset_label,
        "backbone": config.get("backbone", "?"),
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {cls: float(f) for cls, f in zip(CLASSES_5, per_class_f1)},
        "ocular_4class_acc": float(ood_metrics["ocular_4class_subset_acc_pct"]) / 100.0,
        "l2_ood_far_pct": float(ood_metrics["l2_threshold_far_pct"]),
        "l2_ood_rejection_pct": float(ood_metrics["l2_ood_rejection_rate_pct"]),
        "combined_ood_far_pct": float(ood_metrics.get("combined_ood_far_pct", ood_metrics["l2_threshold_far_pct"])),
        "combined_ood_rejection_pct": float(
            ood_metrics.get("combined_ood_rejection_rate_pct", ood_metrics["l2_ood_rejection_rate_pct"])
        ),
        "combined_eye_pass_pct": float(
            ood_metrics.get("combined_eye_pass_rate_pct", ood_metrics["l2_eye_pass_rate_pct"])
        ),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "size_mb": spec.checkpoint.stat().st_size / (1024 * 1024),
        "latency_ms": standalone_ms,
        "combined_latency_ms": standalone_ms + gk_latency_ms,
        "best_epoch": config.get("training_summary", {}).get("best_epoch"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_eye": y_true[eye_mask],
        "y_pred_eye": med_pred_global[eye_mask],
        "test_files": test_files,
    }
    print(
        f"       5-Cls Acc={result['accuracy']*100:.2f}% | MacroF1={result['macro_f1']*100:.2f}% | "
        f"4-Cls EyeAcc={result['ocular_4class_acc']*100:.2f}% | "
        f"2-Layer OOD Rej={result['combined_ood_rejection_pct']:.2f}% (FAR={result['combined_ood_far_pct']:.2f}%) | "
        f"EyePass={result['combined_eye_pass_pct']:.2f}% | Lat={result['latency_ms']:.2f}ms"
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_markdown_report(
    results: list[dict],
    mcnemar_5: Optional[dict],
    mcnemar_eye: Optional[dict],
    gk_info: dict,
) -> str:
    pct = lambda v: f"{v * 100:.2f}%"
    headers = [r["label"] for r in results]
    lines = [
        "# Iris Vision - 5-Class Model Comparison Report (+ Layer 1 Gatekeeper)",
        "",
        f"> Both models are trained and evaluated on the **exact same 5-class unseen test split** of "
        f"`{DATASET_DIR.name}` (100% fundus imagery for all 4 ocular disease classes + `non_fundus` OOD class), "
        f"paired with Layer 1 Binary Gatekeeper (`{gk_info.get('backbone', 'N/A')}`, "
        f"size `{gk_info.get('size_mb', 0.0):.1f} MB`, latency `{gk_info.get('latency_ms', 0.0):.2f} ms`).",
        "",
        "## Overall Metrics",
        "",
        "| Metric | " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join([":---:"] * len(results)) + " |",
    ]

    rows = [
        ("Test dataset", lambda r: r["dataset"]),
        ("Test samples (5-Class)", lambda r: str(r["n_test"])),
        ("Backbone", lambda r: f"`{r['backbone']}`"),
        ("Best epoch", lambda r: str(r["best_epoch"])),
        ("**5-Class Test Accuracy**", lambda r: f"**{pct(r['accuracy'])}**"),
        ("**4-Class Ocular Accuracy (Renorm)**", lambda r: f"**{pct(r['ocular_4class_acc'])}**"),
        ("Macro Precision (5-Class)", lambda r: pct(r["macro_precision"])),
        ("Macro Recall (5-Class)", lambda r: pct(r["macro_recall"])),
        ("**Macro F1 (5-Class)**", lambda r: f"**{pct(r['macro_f1'])}**"),
        ("Weighted F1 (5-Class)", lambda r: pct(r["weighted_f1"])),
    ]
    rows += [(f"F1 `{cls}`", (lambda c: lambda r: f"{r['per_class_f1'][c]:.4f}")(cls)) for cls in CLASSES_5]
    rows += [
        ("Layer-2 OOD Rejection Rate (%)", lambda r: f"{r['l2_ood_rejection_pct']:.2f}%"),
        ("Layer-2 OOD FAR (%)", lambda r: f"{r['l2_ood_far_pct']:.2f}%"),
        ("**2-Layer Combined OOD Rejection (%)**", lambda r: f"**{r['combined_ood_rejection_pct']:.2f}%**"),
        ("**2-Layer Combined OOD FAR (%)**", lambda r: f"**{r['combined_ood_far_pct']:.2f}%**"),
        ("2-Layer Valid Fundus Pass Rate (%)", lambda r: f"{r['combined_eye_pass_pct']:.2f}%"),
        ("Parameters (M)", lambda r: f"{r['params_m']:.2f}"),
        ("Checkpoint size (MB)", lambda r: f"{r['size_mb']:.1f}"),
        ("Model latency (ms/img, batch=1)", lambda r: f"{r['latency_ms']:.2f}"),
        ("2-Layer Pipeline latency (ms/img)", lambda r: f"{r['combined_latency_ms']:.2f}"),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(r) for r in results) + " |")

    if mcnemar_5 is not None:
        a, b = mcnemar_5["labels"]
        verdict = (
            "statistically significant (p < 0.05)"
            if mcnemar_5["p_value"] < 0.05
            else "**not** statistically significant (p >= 0.05)"
        )
        lines += [
            "",
            "## Paired Significance Tests (Identical Test Set)",
            "",
            f"### 1. Full 5-Class Test Set (n={mcnemar_5['n']})",
            f"Exact McNemar test between **{a}** and **{b}**:",
            "",
            f"- Only {a} correct: **{mcnemar_5['a_only_correct']}** images",
            f"- Only {b} correct: **{mcnemar_5['b_only_correct']}** images",
            f"- p-value: **{mcnemar_5['p_value']:.4f}** -> difference is {verdict}.",
        ]

    if mcnemar_eye is not None:
        a, b = mcnemar_eye["labels"]
        verdict_eye = (
            "statistically significant (p < 0.05)"
            if mcnemar_eye["p_value"] < 0.05
            else "**not** statistically significant (p >= 0.05)"
        )
        lines += [
            "",
            f"### 2. Ocular 4-Class Fundus Subset (n={mcnemar_eye['n']})",
            f"Exact McNemar test (renormalized 4-class predictions) between **{a}** and **{b}**:",
            "",
            f"- Only {a} correct: **{mcnemar_eye['a_only_correct']}** images",
            f"- Only {b} correct: **{mcnemar_eye['b_only_correct']}** images",
            f"- p-value: **{mcnemar_eye['p_value']:.4f}** -> difference is {verdict_eye}.",
        ]

    return "\n".join(lines) + "\n"


def plot_comparison(results: list[dict], output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = ["#2A9D8F", "#1D3557"][: len(results)]
    labels = [r["label"] for r in results]

    fig = plt.figure(figsize=(19, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)

    # Panel 1: overall metrics (top-left, 2 cols)
    ax1 = fig.add_subplot(gs[0, 0:2])
    metric_keys = [
        ("accuracy", "5-Cls Acc"),
        ("ocular_4class_acc", "4-Cls Eye"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.72 / len(results)
    for i, r in enumerate(results):
        vals = [r[k] * 100 for k, _ in metric_keys]
        bars = ax1.bar(x + i * width - 0.36 + width / 2, vals, width, label=r["label"], color=palette[i])
        ax1.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax1.set_xticks(x, [n for _, n in metric_keys])
    ax1.set_ylim(75, 102)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Overall 5-Class & 4-Class Accuracy", fontweight="bold")
    ax1.legend(fontsize=8.5, loc="lower right")

    # Panel 2: per-class F1 (top-middle, 2 cols)
    ax2 = fig.add_subplot(gs[0, 2:4])
    x = np.arange(len(CLASSES_5))
    for i, r in enumerate(results):
        vals = [r["per_class_f1"][c] * 100 for c in CLASSES_5]
        bars = ax2.bar(x + i * width - 0.36 + width / 2, vals, width, label=r["label"], color=palette[i])
        ax2.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    ax2.set_xticks(x, [c.replace("_", "\n") for c in CLASSES_5], fontsize=8.5)
    ax2.set_ylim(70, 103)
    ax2.set_ylabel("F1-Score (%)")
    ax2.set_title("Per-Class F1-Score (5 Classes)", fontweight="bold")
    ax2.legend(fontsize=8.5, loc="lower right")

    # Panel 3: efficiency (top-right, 2 cols)
    ax3 = fig.add_subplot(gs[0, 4:6])
    lat = [r["combined_latency_ms"] for r in results]
    size = [r["size_mb"] for r in results]
    xi = np.arange(len(results))
    b1 = ax3.bar(xi - 0.18, lat, 0.34, color=palette, label="2-Layer Latency (ms)")
    ax3.bar_label(b1, fmt="%.1f ms", fontsize=8.5, padding=2)
    ax3.set_ylabel("2-Layer Latency (ms/img)")
    ax3b = ax3.twinx()
    b2 = ax3b.bar(xi + 0.18, size, 0.34, color=palette, alpha=0.45, hatch="//", label="Size (MB)")
    ax3b.bar_label(b2, fmt="%.1f MB", fontsize=8.5, padding=2)
    ax3b.set_ylabel("Checkpoint size (MB)")
    ax3b.grid(False)
    ax3.set_xticks(xi, [l.replace(" (", "\n(") for l in labels], fontsize=9)
    ax3.set_title("Efficiency (solid = 2-Layer latency, hatched = size)", fontweight="bold")

    # Row 2: side-by-side 5x5 normalized confusion matrices (3 cols each)
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i * 3 : (i + 1) * 3])
        cm = r["confusion_matrix"].astype(float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            cbar=False,
            xticklabels=CLASSES_5,
            yticklabels=CLASSES_5,
            ax=ax,
            annot_kws={"size": 9.5, "weight": "bold"},
        )
        ax.set_title(
            f"{r['label']}\n5-Cls Acc {r['accuracy']*100:.2f}% | "
            f"2-Layer OOD Rej {r['combined_ood_rejection_pct']:.2f}% (n={r['n_test']})",
            fontweight="bold",
            fontsize=10.5,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=22, labelsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)

    fig.suptitle(
        "Iris Vision - 5-Class Model Comparison (MobileNetV2 vs ResNet-50 + Layer 1 Gatekeeper)",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Comparison figure saved to: {output_path}")


def run_comparison() -> list[dict]:
    ensure_output_dir()
    device = get_device()

    _, _, test_df = get_5class_split_dataframes(
        dataset_dir=DATASET_DIR,
        balance_train=False,
    )
    test_df = test_df.sort_values("filepath").reset_index(drop=True)
    test_dataset = EyeDiseaseDataset(
        test_df,
        transform=get_default_transforms(is_train=False),
        use_preprocess=True,
        preload_in_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )

    gk_info = {"backbone": "None", "size_mb": 0.0, "latency_ms": 0.0}
    if GATEKEEPER_BEST_PATH.exists():
        gk_model, gk_cfg = load_trained_model(GATEKEEPER_BEST_PATH, GATEKEEPER_CONFIG_JSON, device=device)
        gk_info = {
            "backbone": gk_cfg.get("backbone", "mobilenet_v2"),
            "size_mb": GATEKEEPER_BEST_PATH.stat().st_size / (1024 * 1024),
            "latency_ms": measure_latency_ms(gk_model, device),
        }
        del gk_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    test_files = test_df["filepath"].tolist()
    results = [
        r
        for spec in MODEL_SPECS
        if (r := evaluate_spec(spec, test_loader, test_files, device, gk_info["latency_ms"]))
    ]
    if not results:
        raise RuntimeError("No 5-class model checkpoints found to compare.")

    mcnemar_5 = None
    mcnemar_eye = None
    if len(results) == 2 and results[0]["test_files"] == results[1]["test_files"]:
        a, b = results
        mcnemar_5 = mcnemar_exact(a["y_true"], a["y_pred"], b["y_pred"])
        mcnemar_5["labels"] = (a["label"], b["label"])
        mcnemar_5["n"] = a["n_test"]

        mcnemar_eye = mcnemar_exact(a["y_true_eye"], a["y_pred_eye"], b["y_pred_eye"])
        mcnemar_eye["labels"] = (a["label"], b["label"])
        mcnemar_eye["n"] = int(len(a["y_true_eye"]))

    report = build_markdown_report(results, mcnemar_5, mcnemar_eye, gk_info)
    COMPARISON_REPORT_MD.write_text(report, encoding="utf-8")
    print(f"[SAVED] Comparison report saved to: {COMPARISON_REPORT_MD}")
    plot_comparison(results, COMPARISON_PNG)

    print("\n" + report)
    return results


if __name__ == "__main__":
    run_comparison()
