"""3-way head-to-head model comparison for Iris Vision.

Compares:
1. MobileNetV2 + Old Dataset  (dataset2_balanced)        -> models/baseline_old_dataset/
2. MobileNetV2 + New Dataset  (Retinal Fundus Images)    -> models/mobilenetv2_new_dataset/
3. ResNet-50   + New Dataset  (Retinal Fundus Images)    -> models/resnet50_new_dataset/

Each model is evaluated on the test split of the dataset it was trained on.
Rows 2 vs 3 share the exact same test set (fair architecture comparison) and are
additionally compared with an exact McNemar test.

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
    CLASSES,
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
    PIN_MEMORY,
    USE_AMP,
    ensure_output_dir,
    get_device,
)
from src.dataset import EyeDiseaseDataset, get_default_transforms
from src.evaluate import evaluate_test_set, load_trained_model

COMPARISON_REPORT_MD: Path = OUTPUTS_DIR / "model_comparison_report.md"
COMPARISON_PNG: Path = OUTPUTS_DIR / "model_comparison.png"


@dataclass
class ModelSpec:
    label: str
    dataset_label: str
    checkpoint: Path
    config: Path
    manifest: Path


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        label="MobileNetV2 (Old Dataset)",
        dataset_label="dataset2_balanced",
        checkpoint=MODELS_DIR / "baseline_old_dataset" / "ai_sentinel_mobilenetv2_old.pth",
        config=MODELS_DIR / "baseline_old_dataset" / "model_config.json",
        manifest=OUTPUTS_DIR / "baseline_old_dataset" / "clean_dataset_manifest.csv",
    ),
    ModelSpec(
        label="MobileNetV2 (New Dataset)",
        dataset_label="Retinal Fundus Images",
        checkpoint=MODELS_DIR / "mobilenetv2_new_dataset" / "ai_sentinel_mobilenetv2_new.pth",
        config=MODELS_DIR / "mobilenetv2_new_dataset" / "model_config.json",
        manifest=OUTPUTS_DIR / "clean_dataset_manifest.csv",
    ),
    ModelSpec(
        label="ResNet-50 (New Dataset)",
        dataset_label="Retinal Fundus Images",
        checkpoint=MODELS_DIR / "resnet50_new_dataset" / "ai_sentinel_resnet50_new.pth",
        config=MODELS_DIR / "resnet50_new_dataset" / "model_config.json",
        manifest=OUTPUTS_DIR / "clean_dataset_manifest.csv",
    ),
]


def load_test_df(manifest_path: Path) -> pd.DataFrame:
    """Load the frozen test split from an archived clean manifest."""
    df = pd.read_csv(manifest_path)
    test_df = df[df["orig_split"] == "test"].sort_values("filepath").reset_index(drop=True)

    renamed_old = PROJECT_ROOT / "dataset2_balanced(old dataset, don't use)"
    old_prefix = str(PROJECT_ROOT / "dataset2_balanced")
    if renamed_old.exists():
        test_df["filepath"] = test_df["filepath"].map(
            lambda p: p.replace(old_prefix, str(renamed_old), 1)
            if not Path(p).exists() and p.startswith(old_prefix)
            else p
        )

    missing = (~test_df["filepath"].map(lambda p: Path(p).exists())).sum()
    if missing:
        raise FileNotFoundError(f"{missing} test images listed in {manifest_path} no longer exist.")
    return test_df


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
    device: torch.device,
    dataset_cache: dict[str, EyeDiseaseDataset],
) -> Optional[dict]:
    if not spec.checkpoint.exists():
        print(f"[SKIP] {spec.label}: checkpoint not found at {spec.checkpoint}")
        return None

    print(f"\n[EVAL] {spec.label} ...")
    model, config = load_trained_model(spec.checkpoint, spec.config, device=device)

    test_df = load_test_df(spec.manifest)
    cache_key = "|".join(test_df["filepath"].tolist())
    if cache_key not in dataset_cache:
        dataset_cache[cache_key] = EyeDiseaseDataset(
            test_df,
            transform=get_default_transforms(is_train=False),
            use_preprocess=True,
            preload_in_memory=True,
        )
    loader = DataLoader(
        dataset_cache[cache_key],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )

    y_true, y_pred, _ = evaluate_test_set(model, loader, device=device, use_amp=USE_AMP)
    labels = list(range(len(CLASSES)))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

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
        "per_class_f1": {cls: float(f) for cls, f in zip(CLASSES, per_class_f1)},
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "size_mb": spec.checkpoint.stat().st_size / (1024 * 1024),
        "latency_ms": measure_latency_ms(model, device),
        "best_epoch": config.get("training_summary", {}).get("best_epoch"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "y_true": y_true,
        "y_pred": y_pred,
        "test_files": test_df["filepath"].tolist(),
    }
    print(
        f"       Acc={result['accuracy']*100:.2f}% | MacroF1={result['macro_f1']*100:.2f}% | "
        f"Latency={result['latency_ms']:.2f} ms | Size={result['size_mb']:.1f} MB"
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_markdown_report(results: list[dict], mcnemar: Optional[dict]) -> str:
    pct = lambda v: f"{v * 100:.2f}%"
    headers = [r["label"] for r in results]
    lines = [
        "# Iris Vision - 3-Way Model Comparison Report",
        "",
        "> Each model is evaluated on the **test split of the dataset it was trained on**. "
        "The two *New Dataset* models share the exact same test set, so their comparison is a "
        "fair architecture comparison. The *Old Dataset* row reflects a different (anterior-segment "
        "cataract) test set and is a dataset-level reference, not a like-for-like benchmark.",
        "",
        "## Overall Metrics",
        "",
        "| Metric | " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join([":---:"] * len(results)) + " |",
    ]

    rows = [
        ("Test dataset", lambda r: r["dataset"]),
        ("Test samples", lambda r: str(r["n_test"])),
        ("Backbone", lambda r: f"`{r['backbone']}`"),
        ("Best epoch", lambda r: str(r["best_epoch"])),
        ("**Test Accuracy**", lambda r: f"**{pct(r['accuracy'])}**"),
        ("Macro Precision", lambda r: pct(r["macro_precision"])),
        ("Macro Recall", lambda r: pct(r["macro_recall"])),
        ("**Macro F1**", lambda r: f"**{pct(r['macro_f1'])}**"),
        ("Weighted F1", lambda r: pct(r["weighted_f1"])),
    ]
    rows += [(f"F1 `{cls}`", (lambda c: lambda r: f"{r['per_class_f1'][c]:.4f}")(cls)) for cls in CLASSES]
    rows += [
        ("Parameters (M)", lambda r: f"{r['params_m']:.2f}"),
        ("Checkpoint size (MB)", lambda r: f"{r['size_mb']:.1f}"),
        ("GPU latency (ms/img, batch=1)", lambda r: f"{r['latency_ms']:.2f}"),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(r) for r in results) + " |")

    if mcnemar is not None:
        a, b = mcnemar["labels"]
        verdict = (
            "statistically significant (p < 0.05)"
            if mcnemar["p_value"] < 0.05
            else "**not** statistically significant (p >= 0.05)"
        )
        lines += [
            "",
            "## Paired Significance Test (Same Test Set)",
            "",
            f"Exact McNemar test between **{a}** and **{b}** on {mcnemar['n']} identical test images:",
            "",
            f"- Only {a} correct: **{mcnemar['a_only_correct']}** images",
            f"- Only {b} correct: **{mcnemar['b_only_correct']}** images",
            f"- p-value: **{mcnemar['p_value']:.4f}** -> difference is {verdict}.",
        ]

    return "\n".join(lines) + "\n"


def plot_comparison(results: list[dict], output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = ["#94A3B8", "#2A9D8F", "#E76F51"][: len(results)]
    labels = [r["label"] for r in results]

    fig = plt.figure(figsize=(20, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, max(3, len(results)))

    # Panel 1: overall metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metric_keys = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        vals = [r[k] * 100 for k, _ in metric_keys]
        bars = ax1.bar(x + i * width - 0.4 + width / 2, vals, width, label=r["label"], color=palette[i])
        ax1.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax1.set_xticks(x, [n for _, n in metric_keys])
    ax1.set_ylim(80, 101)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Overall Test Metrics", fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")

    # Panel 2: per-class F1
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(CLASSES))
    for i, r in enumerate(results):
        vals = [r["per_class_f1"][c] * 100 for c in CLASSES]
        bars = ax2.bar(x + i * width - 0.4 + width / 2, vals, width, label=r["label"], color=palette[i])
        ax2.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax2.set_xticks(x, [c.replace("_", "\n") for c in CLASSES])
    ax2.set_ylim(70, 102)
    ax2.set_ylabel("F1-Score (%)")
    ax2.set_title("Per-Class F1-Score", fontweight="bold")

    # Panel 3: efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    lat = [r["latency_ms"] for r in results]
    size = [r["size_mb"] for r in results]
    xi = np.arange(len(results))
    b1 = ax3.bar(xi - 0.2, lat, 0.4, color=palette, label="Latency (ms/img)")
    ax3.bar_label(b1, fmt="%.1f ms", fontsize=8, padding=2)
    ax3.set_ylabel("GPU latency (ms/img)")
    ax3b = ax3.twinx()
    b2 = ax3b.bar(xi + 0.2, size, 0.4, color=palette, alpha=0.45, hatch="//", label="Size (MB)")
    ax3b.bar_label(b2, fmt="%.0f MB", fontsize=8, padding=2)
    ax3b.set_ylabel("Checkpoint size (MB)")
    ax3b.grid(False)
    ax3.set_xticks(xi, [l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax3.set_title("Efficiency (solid = latency, hatched = size)", fontweight="bold")

    # Row 2: normalized confusion matrices
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        cm = r["confusion_matrix"].astype(float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        sns.heatmap(
            cm_norm, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, cbar=False,
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, annot_kws={"size": 10, "weight": "bold"},
        )
        ax.set_title(f"{r['label']}\nAcc {r['accuracy']*100:.2f}% (n={r['n_test']})", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Iris Vision - 3-Way Model Comparison", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Comparison figure saved to: {output_path}")


def run_comparison() -> list[dict]:
    ensure_output_dir()
    device = get_device()
    dataset_cache: dict[str, EyeDiseaseDataset] = {}

    results = [r for spec in MODEL_SPECS if (r := evaluate_spec(spec, device, dataset_cache))]
    if not results:
        raise RuntimeError("No model checkpoints found to compare.")

    # Paired McNemar between the two New Dataset models (identical test files)
    mcnemar = None
    new_models = [r for r in results if r["dataset"] == "Retinal Fundus Images"]
    if len(new_models) == 2 and new_models[0]["test_files"] == new_models[1]["test_files"]:
        a, b = new_models
        mcnemar = mcnemar_exact(a["y_true"], a["y_pred"], b["y_pred"])
        mcnemar["labels"] = (a["label"], b["label"])
        mcnemar["n"] = a["n_test"]

    report = build_markdown_report(results, mcnemar)
    COMPARISON_REPORT_MD.write_text(report, encoding="utf-8")
    print(f"[SAVED] Comparison report saved to: {COMPARISON_REPORT_MD}")
    plot_comparison(results, COMPARISON_PNG)

    print("\n" + report)
    return results


if __name__ == "__main__":
    run_comparison()

