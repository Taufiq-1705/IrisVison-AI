"""Two-phase GPU training pipeline for Iris Vision on Windows 11 Native.

Phase 1: Head warm-up (backbone frozen, 4 epochs, lr=1e-3).
Phase 2: Full-network fine-tuning (16 epochs, differential lr=1e-4 / 3e-4,
         CosineAnnealingLR, EarlyStopping on Validation Macro F1-Score).
Uses PyTorch AMP (`torch.amp.autocast('cuda')` & `GradScaler`) and weighted
`CrossEntropyLoss(label_smoothing=0.05)`.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running directly as `python src/train.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BACKBONE_LR,
    BACKBONE_NAME,
    BATCH_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CLASS_TO_IDX,
    CLASSES,
    DATASET_DIR,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_EPOCHS,
    HEAD_LR,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_SMOOTHING,
    MODEL_BEST_PATH,
    MODEL_CONFIG_JSON,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    TRAINING_HISTORY_JSON,
    USE_AMP,
    WARMUP_EPOCHS,
    WARMUP_LR,
    WEIGHT_DECAY,
    ensure_model_dir,
    ensure_output_dir,
    get_device,
)
from src.dataset import (
    EyeDiseaseDataset,
    compute_class_weights,
    get_default_transforms,
    get_split_dataframes,
)
from src.model import build_model


def seed_everything(seed: int = SEED) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Run one training epoch with AMP and return (loss, accuracy, f1_macro)."""
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    total_samples = len(all_labels)
    avg_loss = running_loss / max(1, total_samples)
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, f1


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Evaluate model on validation split and return (loss, accuracy, f1_macro)."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total_samples = len(all_labels)
    avg_loss = running_loss / max(1, total_samples)
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, f1


def save_best_artifacts(
    model: nn.Module,
    backbone_name: str,
    best_epoch: int,
    best_phase: str,
    best_val_f1: float,
    best_val_acc: float,
    best_val_loss: float,
    warmup_epochs: int,
    finetune_epochs: int,
    class_weights: torch.Tensor,
) -> None:
    """Save best model weights (.pth) and config metadata (.json)."""
    ensure_model_dir()

    checkpoint = {
        "epoch": best_epoch,
        "phase": best_phase,
        "backbone_name": backbone_name,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "model_state_dict": model.state_dict(),
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, MODEL_BEST_PATH)

    weights_init = (
        "IMAGENET1K_V2" if backbone_name in ("mobilenet_v2", "resnet50") else "IMAGENET1K_V1"
    )
    config_payload = {
        "project": "Iris Vision",
        "backbone": backbone_name,
        "weights_init": weights_init,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
        "image_size": list(IMAGE_SIZE),
        "preprocessing": {
            "safe_crop_fundus_roi": True,
            "clahe_lab": True,
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
        },
        "training_summary": {
            "best_epoch": best_epoch,
            "best_phase": best_phase,
            "best_val_f1_macro": round(best_val_f1, 4),
            "best_val_accuracy": round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "warmup_epochs": warmup_epochs,
            "finetune_epochs": finetune_epochs,
            "label_smoothing": LABEL_SMOOTHING,
            "class_weights": [round(float(w), 4) for w in class_weights.cpu().tolist()],
            "clean_conflicting_duplicates": True,
        },
    }
    with open(MODEL_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)


def run_training(
    backbone_name: str = BACKBONE_NAME,
    warmup_epochs: int = WARMUP_EPOCHS,
    finetune_epochs: int = FINETUNE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = SEED,
) -> Dict:
    """Execute full 2-phase GPU training and save best model & training history."""
    seed_everything(seed)
    ensure_output_dir()
    ensure_model_dir()

    device = get_device()
    use_amp = (device.type == "cuda") and USE_AMP
    print("=" * 72)
    print(f"IRIS VISION - TWO-PHASE TRAINING ({backbone_name.upper()})")
    print(f"Device: {device} | AMP Enabled: {use_amp} | Batch Size: {batch_size}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    # 1. Load cleaned splits (conflicting MD5 duplicates pruned)
    train_df, val_df, test_df = get_split_dataframes(
        dataset_dir=DATASET_DIR,
        use_existing_splits=True,
        clean_conflicting_duplicates=True,
        balance_train=False,
        seed=seed,
    )
    print(f"Train samples: {len(train_df)} | Class counts: {train_df['class_name'].value_counts().to_dict()}")
    print(f"Val samples  : {len(val_df)} | Class counts: {val_df['class_name'].value_counts().to_dict()}")
    print(f"Test samples : {len(test_df)} | Class counts: {test_df['class_name'].value_counts().to_dict()}")

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

    effective_workers = 0
    persistent = False
    pin = PIN_MEMORY and (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    # 2. Weighted CrossEntropyLoss with Label Smoothing
    class_weights = compute_class_weights(train_df).to(device)
    print(f"Class Weights: {[round(w, 4) for w in class_weights.cpu().tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # 3. Build Model & AMP Scaler
    model = build_model(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=True).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_phase = "phase1_warmup"
    global_epoch = 0

    # ==========================================================================
    # PHASE 1: Head Warm-up (Backbone Frozen)
    # ==========================================================================
    print(f"\n--- Phase 1: Head Warm-up ({warmup_epochs} epochs, lr={WARMUP_LR}) ---")
    model.freeze_backbone()
    optimizer_p1 = torch.optim.AdamW(
        model.get_head_params(), lr=WARMUP_LR, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, warmup_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer_p1, scaler, device, use_amp
        )
        va_loss, va_acc, va_f1 = validate_one_epoch(
            model, val_loader, criterion, device, use_amp
        )
        elapsed = time.time() - t0

        improved = va_f1 > best_val_f1
        if improved:
            best_val_f1 = va_f1
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = global_epoch
            best_phase = "phase1_warmup"
            save_best_artifacts(
                model,
                backbone_name,
                best_epoch,
                best_phase,
                best_val_f1,
                best_val_acc,
                best_val_loss,
                warmup_epochs,
                finetune_epochs,
                class_weights,
            )

        mark = " *" if improved else ""
        print(
            f"[P1 Epoch {epoch:02d}/{warmup_epochs:02d} | G{global_epoch:02d}] "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} "
            f"({elapsed:.1f}s){mark}"
        )

        history.append(
            {
                "epoch": global_epoch,
                "phase": "warmup",
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 5),
                "train_f1": round(tr_f1, 5),
                "val_loss": round(va_loss, 5),
                "val_acc": round(va_acc, 5),
                "val_f1": round(va_f1, 5),
                "lr_backbone": 0.0,
                "lr_head": WARMUP_LR,
            }
        )

    # ==========================================================================
    # PHASE 2: End-to-End Fine-Tuning (Differential LR + CosineAnnealingLR)
    # ==========================================================================
    print(
        f"\n--- Phase 2: Full Fine-Tuning ({finetune_epochs} epochs, "
        f"backbone_lr={BACKBONE_LR}, head_lr={HEAD_LR}, patience={patience}) ---"
    )
    model.unfreeze_backbone()
    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": model.get_backbone_params(), "lr": BACKBONE_LR},
            {"params": model.get_head_params(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=finetune_epochs, eta_min=1e-6
    )

    no_improve_count = 0
    for epoch in range(1, finetune_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer_p2, scaler, device, use_amp
        )
        va_loss, va_acc, va_f1 = validate_one_epoch(
            model, val_loader, criterion, device, use_amp
        )
        lr_bb = float(optimizer_p2.param_groups[0]["lr"])
        lr_hd = float(optimizer_p2.param_groups[1]["lr"])
        scheduler_p2.step()
        elapsed = time.time() - t0

        improved = va_f1 > (best_val_f1 + 1e-4)
        if improved:
            best_val_f1 = va_f1
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = global_epoch
            best_phase = "phase2_finetune"
            no_improve_count = 0
            save_best_artifacts(
                model,
                backbone_name,
                best_epoch,
                best_phase,
                best_val_f1,
                best_val_acc,
                best_val_loss,
                warmup_epochs,
                finetune_epochs,
                class_weights,
            )
        else:
            no_improve_count += 1

        mark = " *" if improved else ""
        print(
            f"[P2 Epoch {epoch:02d}/{finetune_epochs:02d} | G{global_epoch:02d}] "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} "
            f"| LR: {lr_bb:.1e}/{lr_hd:.1e} ({elapsed:.1f}s){mark}"
        )

        history.append(
            {
                "epoch": global_epoch,
                "phase": "finetune",
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 5),
                "train_f1": round(tr_f1, 5),
                "val_loss": round(va_loss, 5),
                "val_acc": round(va_acc, 5),
                "val_f1": round(va_f1, 5),
                "lr_backbone": lr_bb,
                "lr_head": lr_hd,
            }
        )

        if no_improve_count >= patience:
            print(
                f"\n[EARLY STOPPING] No improvement in Val F1 for {patience} epochs "
                f"(Best Val F1={best_val_f1:.4f} at Epoch {best_epoch})."
            )
            break

    # Save training history
    with open(TRAINING_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": backbone_name,
                "best_epoch": best_epoch,
                "best_val_f1": round(best_val_f1, 5),
                "best_val_acc": round(best_val_acc, 5),
                "warmup_epochs": warmup_epochs,
                "history": history,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print(f"[DONE] Best Model Saved  : {MODEL_BEST_PATH}")
    print(f"[DONE] Model Config Saved: {MODEL_CONFIG_JSON}")
    print(f"[DONE] History Saved     : {TRAINING_HISTORY_JSON}")
    print(
        f"Best Epoch: {best_epoch} ({best_phase}) | "
        f"Best Val Acc: {best_val_acc:.4f} | Best Val F1: {best_val_f1:.4f}"
    )
    print("=" * 72)

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Iris Vision classification model.")
    parser.add_argument(
        "--backbone",
        type=str,
        default=BACKBONE_NAME,
        choices=["mobilenet_v2", "efficientnet_b0", "resnet50"],
        help="Backbone architecture (default: mobilenet_v2)",
    )
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        backbone_name=args.backbone,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patience=args.patience,
    )

