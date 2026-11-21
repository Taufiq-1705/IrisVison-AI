"""Layer 1 Binary Gatekeeper training & evaluation pipeline for AI Sentinel.

Trains a lightweight MobileNetV2 (`IMAGENET1K_V2`) binary classifier:
- Class 0 (`non_eye`)   : Out-of-distribution / non-fundus images (`non_fundus`)
- Class 1 (`valid_eye`) : Valid ocular images (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`)

Uses PyTorch AMP FP16 (`torch.amp.autocast('cuda')`), balanced `CrossEntropyLoss`,
and evaluates both standard argmax (0.50) and the production gatekeeper threshold
(`P(valid_eye) >= 0.65`).
Saves best checkpoint to `models/gatekeeper_best.pth` and test confusion matrix to
`outputs/gatekeeper_confusion_matrix.png`.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    DATASET_DIR,
    GATEKEEPER_BEST_PATH,
    GATEKEEPER_CLASS_TO_IDX,
    GATEKEEPER_CLASSES,
    GATEKEEPER_CONFUSION_MATRIX_PNG,
    GATEKEEPER_CONFIG_JSON,
    GATEKEEPER_CURVES_PNG,
    GATEKEEPER_EYE_THRESHOLD,
    GATEKEEPER_HISTORY_JSON,
    GATEKEEPER_IDX_TO_CLASS,
    IMAGE_SIZE,
    PIN_MEMORY,
    SEED,
    USE_AMP,
    WEIGHT_DECAY,
    ensure_model_dir,
    ensure_output_dir,
    get_device,
)
from src.dataset import (
    EyeDiseaseDataset,
    compute_class_weights,
    get_default_transforms,
    get_gatekeeper_split_dataframes,
)
from src.model import build_model
from src.train import seed_everything, train_one_epoch


@torch.no_grad()
def evaluate_gatekeeper(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    eye_threshold: float = GATEKEEPER_EYE_THRESHOLD,
) -> Dict[str, float | np.ndarray]:
    """Evaluate binary gatekeeper at both argmax (0.50) and custom `eye_threshold` (0.65)."""
    model.eval()
    running_loss = 0.0
    all_labels: list[int] = []
    all_eye_probs: list[float] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1)

        running_loss += float(loss.item()) * images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_eye_probs.extend(probs[:, 1].cpu().tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    p_eye = np.array(all_eye_probs, dtype=np.float64)
    y_pred_thresh = (p_eye >= eye_threshold).astype(np.int64)

    avg_loss = running_loss / max(1, len(y_true))
    acc = float(accuracy_score(y_true, y_pred_thresh))
    macro_f1 = float(f1_score(y_true, y_pred_thresh, average="macro", zero_division=0))
    recalls = recall_score(y_true, y_pred_thresh, labels=[0, 1], average=None, zero_division=0)
    recall_non_eye = float(recalls[0])
    recall_valid_eye = float(recalls[1])
    try:
        auc = float(roc_auc_score(y_true, p_eye))
    except ValueError:
        auc = 1.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "recall_non_eye": recall_non_eye,
        "recall_valid_eye": recall_valid_eye,
        "roc_auc": auc,
        "y_true": y_true,
        "y_pred": y_pred_thresh,
        "p_eye": p_eye,
    }


def plot_gatekeeper_results(
    history: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    curves_path: Path = GATEKEEPER_CURVES_PNG,
    cm_path: Path = GATEKEEPER_CONFUSION_MATRIX_PNG,
) -> None:
    """Save training curves and test confusion matrix for the binary Gatekeeper."""
    df_h = pd.DataFrame(history)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axes[0].plot(df_h["epoch"], df_h["train_loss"], marker="o", label="Train Loss", color="#264653")
    axes[0].plot(df_h["epoch"], df_h["val_loss"], marker="s", label="Val Loss", color="#E76F51")
    axes[0].set_title("Gatekeeper Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(df_h["epoch"], np.array(df_h["val_recall_valid_eye"]) * 100, marker="o", label="Val Recall: valid_eye (%)", color="#2A9D8F")
    axes[1].plot(df_h["epoch"], np.array(df_h["val_recall_non_eye"]) * 100, marker="s", label="Val Recall: non_eye (%)", color="#E63946")
    axes[1].axhline(99.0, color="#2A9D8F", linestyle="--", alpha=0.6, label="Target valid_eye (99%)")
    axes[1].axhline(98.0, color="#E63946", linestyle=":", alpha=0.6, label="Target non_eye (98%)")
    axes[1].set_title(f"Gatekeeper Validation Recalls (Threshold={GATEKEEPER_EYE_THRESHOLD})", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall (%)")
    axes[1].legend(fontsize=8.5)

    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GATEKEEPER_CLASSES,
        yticklabels=GATEKEEPER_CLASSES,
        cbar=False,
        ax=axes[0],
        annot_kws={"size": 12, "weight": "bold"},
    )
    axes[0].set_title("Gatekeeper Test Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=GATEKEEPER_CLASSES,
        yticklabels=GATEKEEPER_CLASSES,
        vmin=0.0,
        vmax=100.0,
        ax=axes[1],
        annot_kws={"size": 12, "weight": "bold"},
    )
    axes[1].set_title("Gatekeeper Test Confusion Matrix (Recall %)", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.suptitle(
        f"Layer 1 Binary Gatekeeper (MobileNetV2) | Threshold P(valid_eye) >= {GATEKEEPER_EYE_THRESHOLD}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_gatekeeper_training(
    backbone_name: str = "mobilenet_v2",
    warmup_epochs: int = 2,
    finetune_epochs: int = 8,
    batch_size: int = BATCH_SIZE,
    eye_threshold: float = GATEKEEPER_EYE_THRESHOLD,
    seed: int = SEED,
) -> Dict:
    """Train and evaluate Layer 1 binary Gatekeeper on GPU with AMP FP16."""
    seed_everything(seed)
    ensure_output_dir()
    ensure_model_dir()

    device = get_device()
    use_amp = (device.type == "cuda") and USE_AMP
    total_epochs = warmup_epochs + finetune_epochs

    print("=" * 76)
    print(f"AI SENTINEL - LAYER 1 BINARY GATEKEEPER TRAINING ({backbone_name.upper()})")
    print(f"Device: {device} | AMP: {use_amp} | Epochs: {total_epochs} ({warmup_epochs} W + {finetune_epochs} FT)")
    print(f"Operational Gatekeeper Threshold: P(valid_eye) >= {eye_threshold:.2f}")
    print("=" * 76)

    train_df, val_df, test_df = get_gatekeeper_split_dataframes(
        dataset_dir=DATASET_DIR,
        balance_train=False,
        seed=seed,
    )
    print(f"Train samples: {len(train_df)} | Counts: {train_df['class_name'].value_counts().to_dict()}")
    print(f"Val samples  : {len(val_df)} | Counts: {val_df['class_name'].value_counts().to_dict()}")
    print(f"Test samples : {len(test_df)} | Counts: {test_df['class_name'].value_counts().to_dict()}")

    train_dataset = EyeDiseaseDataset(
        train_df,
        transform=get_default_transforms(is_train=True),
        use_preprocess=True,
        preload_in_memory=True,
    )
    val_dataset = EyeDiseaseDataset(
        val_df,
        transform=get_default_transforms(is_train=False),
        use_preprocess=True,
        preload_in_memory=True,
    )
    test_dataset = EyeDiseaseDataset(
        test_df,
        transform=get_default_transforms(is_train=False),
        use_preprocess=True,
        preload_in_memory=True,
    )

    pin = PIN_MEMORY and (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # Balanced class weights with slight protection on valid_eye so valid_eye recall >= 99% at threshold 0.65
    class_weights = compute_class_weights(train_df, num_classes=2).to(device)
    class_weights[1] = class_weights[1] * 1.25  # Extra protection against false rejection of real eyes
    class_weights = class_weights / class_weights.mean()
    print(f"Binary Class Weights (0=non_eye, 1=valid_eye): {[round(float(w), 4) for w in class_weights.cpu().tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    model = build_model(backbone_name=backbone_name, num_classes=2, pretrained=True).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_score = -1.0
    best_epoch = 0
    global_epoch = 0

    # Phase 1: Warm-up classification head
    model.freeze_backbone()
    opt_p1 = torch.optim.AdamW(model.get_head_params(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    for ep in range(1, warmup_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, opt_p1, scaler, device, use_amp)
        val_metrics = evaluate_gatekeeper(model, val_loader, criterion, device, use_amp, eye_threshold=eye_threshold)
        elapsed = time.time() - t0

        # Composite score prioritizing valid_eye recall >= 99% and non_eye recall >= 98%
        rec_eye = float(val_metrics["recall_valid_eye"])
        rec_non = float(val_metrics["recall_non_eye"])
        val_f1 = float(val_metrics["macro_f1"])
        score = 0.50 * rec_eye + 0.35 * rec_non + 0.15 * val_f1

        mark = ""
        if score > best_score:
            best_score = score
            best_epoch = global_epoch
            mark = " *"
            _save_gatekeeper_checkpoint(model, backbone_name, best_epoch, val_metrics, eye_threshold)

        print(
            f"[GK-P1 {ep:02d}/{warmup_epochs:02d} | E{global_epoch:02d}] "
            f"TrLoss: {tr_loss:.4f} | ValLoss: {val_metrics['loss']:.4f} "
            f"Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Rec(valid_eye): {rec_eye*100:.2f}% | Rec(non_eye): {rec_non*100:.2f}% ({elapsed:.1f}s){mark}"
        )
        history.append(
            {
                "epoch": global_epoch,
                "phase": "warmup",
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 5),
                "val_loss": round(float(val_metrics["loss"]), 5),
                "val_acc": round(float(val_metrics["accuracy"]), 5),
                "val_f1": round(val_f1, 5),
                "val_recall_valid_eye": round(rec_eye, 5),
                "val_recall_non_eye": round(rec_non, 5),
            }
        )

    # Phase 2: Full fine-tuning
    model.unfreeze_backbone()
    opt_p2 = torch.optim.AdamW(
        [
            {"params": model.get_backbone_params(), "lr": 1e-4},
            {"params": model.get_head_params(), "lr": 3e-4},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    sched_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=finetune_epochs, eta_min=1e-6)

    for ep in range(1, finetune_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, opt_p2, scaler, device, use_amp)
        val_metrics = evaluate_gatekeeper(model, val_loader, criterion, device, use_amp, eye_threshold=eye_threshold)
        sched_p2.step()
        elapsed = time.time() - t0

        rec_eye = float(val_metrics["recall_valid_eye"])
        rec_non = float(val_metrics["recall_non_eye"])
        val_f1 = float(val_metrics["macro_f1"])
        score = 0.50 * rec_eye + 0.35 * rec_non + 0.15 * val_f1

        mark = ""
        if score >= best_score:
            best_score = score
            best_epoch = global_epoch
            mark = " *"
            _save_gatekeeper_checkpoint(model, backbone_name, best_epoch, val_metrics, eye_threshold)

        print(
            f"[GK-P2 {ep:02d}/{finetune_epochs:02d} | E{global_epoch:02d}] "
            f"TrLoss: {tr_loss:.4f} | ValLoss: {val_metrics['loss']:.4f} "
            f"Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Rec(valid_eye): {rec_eye*100:.2f}% | Rec(non_eye): {rec_non*100:.2f}% ({elapsed:.1f}s){mark}"
        )
        history.append(
            {
                "epoch": global_epoch,
                "phase": "finetune",
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 5),
                "val_loss": round(float(val_metrics["loss"]), 5),
                "val_acc": round(float(val_metrics["accuracy"]), 5),
                "val_f1": round(val_f1, 5),
                "val_recall_valid_eye": round(rec_eye, 5),
                "val_recall_non_eye": round(rec_non, 5),
            }
        )

    with open(GATEKEEPER_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump({"backbone": backbone_name, "best_epoch": best_epoch, "history": history}, f, indent=2)

    # Evaluate best checkpoint on Test Set
    best_ckpt = torch.load(GATEKEEPER_BEST_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_gatekeeper(model, test_loader, criterion, device, use_amp, eye_threshold=eye_threshold)

    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]
    rec_valid_eye = float(test_metrics["recall_valid_eye"]) * 100.0
    rec_non_eye = float(test_metrics["recall_non_eye"]) * 100.0
    test_acc = float(test_metrics["accuracy"]) * 100.0
    test_f1 = float(test_metrics["macro_f1"]) * 100.0
    test_auc = float(test_metrics["roc_auc"]) * 100.0

    print("\n" + "=" * 76)
    print(f"LAYER 1 GATEKEEPER - TEST SET EVALUATION (Best Epoch {best_epoch})")
    print("=" * 76)
    print(f"Test Samples               : {len(test_df)} (valid_eye={(y_true==1).sum()}, non_eye={(y_true==0).sum()})")
    print(f"Gatekeeper Threshold       : P(valid_eye) >= {eye_threshold:.2f}")
    print(f"Test Accuracy              : {test_acc:.2f}%")
    print(f"Test Macro F1-Score        : {test_f1:.2f}%")
    print(f"Test ROC-AUC               : {test_auc:.2f}%")
    print(f"Recall `valid_eye` (Target >= 99.0%): {rec_valid_eye:.2f}%")
    print(f"Recall `non_eye`   (Target >= 98.0%): {rec_non_eye:.2f}%")
    print("\n--- Gatekeeper Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=GATEKEEPER_CLASSES, digits=4, zero_division=0))
    print("=" * 76)

    plot_gatekeeper_results(history, y_true, y_pred)
    print(f"[SAVED] Best Gatekeeper Checkpoint : {GATEKEEPER_BEST_PATH}")
    print(f"[SAVED] Gatekeeper Config JSON     : {GATEKEEPER_CONFIG_JSON}")
    print(f"[SAVED] Gatekeeper Confusion Matrix: {GATEKEEPER_CONFUSION_MATRIX_PNG}")

    return {
        "best_epoch": best_epoch,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_recall_valid_eye": rec_valid_eye,
        "test_recall_non_eye": rec_non_eye,
        "test_roc_auc": test_auc,
    }


def _save_gatekeeper_checkpoint(
    model: nn.Module,
    backbone_name: str,
    best_epoch: int,
    val_metrics: Dict,
    eye_threshold: float,
) -> None:
    ckpt = {
        "epoch": best_epoch,
        "backbone_name": backbone_name,
        "num_classes": 2,
        "classes": GATEKEEPER_CLASSES,
        "class_to_idx": GATEKEEPER_CLASS_TO_IDX,
        "idx_to_class": GATEKEEPER_IDX_TO_CLASS,
        "eye_threshold": eye_threshold,
        "model_state_dict": model.state_dict(),
        "best_val_f1": float(val_metrics["macro_f1"]),
        "best_val_acc": float(val_metrics["accuracy"]),
        "best_val_recall_valid_eye": float(val_metrics["recall_valid_eye"]),
        "best_val_recall_non_eye": float(val_metrics["recall_non_eye"]),
    }
    torch.save(ckpt, GATEKEEPER_BEST_PATH)

    cfg = {
        "project": "Iris Vision - Layer 1 Gatekeeper",
        "backbone": backbone_name,
        "num_classes": 2,
        "classes": GATEKEEPER_CLASSES,
        "class_to_idx": GATEKEEPER_CLASS_TO_IDX,
        "idx_to_class": {str(k): v for k, v in GATEKEEPER_IDX_TO_CLASS.items()},
        "image_size": list(IMAGE_SIZE),
        "eye_threshold": eye_threshold,
        "validation_summary": {
            "best_epoch": best_epoch,
            "val_accuracy": round(float(val_metrics["accuracy"]), 4),
            "val_macro_f1": round(float(val_metrics["macro_f1"]), 4),
            "val_recall_valid_eye": round(float(val_metrics["recall_valid_eye"]), 4),
            "val_recall_non_eye": round(float(val_metrics["recall_non_eye"]), 4),
        },
    }
    with open(GATEKEEPER_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Layer 1 Binary Gatekeeper.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "mobilenet_v3_small"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=GATEKEEPER_EYE_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gatekeeper_training(
        backbone_name=args.backbone,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        eye_threshold=args.threshold,
    )
