"""Model evaluation, training curves, 5x5 confusion matrix, OOD FAR, and Grad-CAM for Iris Vision."""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
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
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BACKBONE_NAME,
    BATCH_SIZE,
    CLASSES,
    CLASSES_5,
    CONFUSION_MATRIX_5CLASS_PNG,
    CONFUSION_MATRIX_PNG,
    DATASET_DIR,
    GATEKEEPER_BEST_PATH,
    GATEKEEPER_CONFIG_JSON,
    GATEKEEPER_EYE_THRESHOLD,
    GRADCAM_5CLASS_PNG,
    GRADCAM_SAMPLES_PNG,
    IMAGE_SIZE,
    L2_NON_FUNDUS_THRESHOLD,
    MODEL_5CLASS_PATH,
    MODEL_BEST_PATH,
    MODEL_CONFIG_JSON,
    NUM_CLASSES,
    OOD_CLASS_NAME,
    PIN_MEMORY,
    TRAINING_CURVES_5CLASS_PNG,
    TRAINING_CURVES_PNG,
    TRAINING_HISTORY_5CLASS_JSON,
    TRAINING_HISTORY_JSON,
    USE_AMP,
    ensure_output_dir,
    get_device,
)
from src.dataset import (
    EyeDiseaseDataset,
    get_5class_split_dataframes,
    get_default_transforms,
    get_split_dataframes,
)
from src.model import build_model
from src.preprocess import preprocess_image


class GradCAM:
    """Native PyTorch Grad-CAM extractor compatible with inplace ReLU activations."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output: torch.Tensor) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """Compute normalized 2D Grad-CAM heatmap in [0, 1] and prediction details."""
        self.model.eval()
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_conf = float(probs[0, pred_idx].item())
        target_cls = pred_idx if class_idx is None else int(class_idx)

        self.model.zero_grad(set_to_none=True)
        score = logits[0, target_cls]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        cam_resized = cv2.resize(cam_np, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)
        return cam_resized, pred_idx, pred_conf

    def close(self) -> None:
        self._forward_handle.remove()


def load_trained_model(
    checkpoint_path: Path = MODEL_BEST_PATH,
    config_path: Path = MODEL_CONFIG_JSON,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, dict]:
    """Load trained model checkpoint and configuration JSON."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone_name = checkpoint.get("backbone_name", BACKBONE_NAME)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    ckpt_classes = checkpoint.get(
        "classes",
        CLASSES_5 if num_classes == 5 else CLASSES,
    )

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    config["backbone"] = backbone_name
    config["num_classes"] = num_classes
    config["classes"] = ckpt_classes

    model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on test_loader and return (y_true, y_pred, y_probs)."""
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    amp_enabled = (device.type == "cuda") and use_amp
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            probs = F.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_probs = np.concatenate(all_probs, axis=0)
    return y_true, y_pred, y_probs


def plot_training_curves(
    history_path: Path = TRAINING_HISTORY_JSON,
    output_path: Path = TRAINING_CURVES_PNG,
) -> None:
    """Plot Loss, Accuracy, and Macro F1-Score across Phase 1 and Phase 2."""
    if not history_path.exists():
        print(f"[WARN] Training history not found at {history_path}. Skipping curves plot.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    history = payload.get("history", [])
    if not history:
        return

    df_hist = pd.DataFrame(history)
    warmup_epochs = payload.get("warmup_epochs", 4)
    best_epoch = payload.get("best_epoch", None)
    num_cls = payload.get("num_classes", 4)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    metrics = [
        ("train_loss", "val_loss", "Cross-Entropy Loss", "Loss"),
        ("train_acc", "val_acc", "Accuracy", "Accuracy"),
        ("train_f1", "val_f1", "Macro F1-Score", "F1-Score"),
    ]

    for ax, (tr_col, va_col, title, ylabel) in zip(axes, metrics):
        ax.plot(df_hist["epoch"], df_hist[tr_col], marker="o", markersize=4, linewidth=2, label="Train", color="#264653")
        ax.plot(df_hist["epoch"], df_hist[va_col], marker="s", markersize=4, linewidth=2, label="Validation", color="#E76F51")

        if df_hist["epoch"].max() > warmup_epochs:
            ax.axvline(warmup_epochs + 0.5, color="#457B9D", linestyle="--", linewidth=1.4, label="Phase 2 Start")

        if best_epoch is not None and best_epoch in df_hist["epoch"].values:
            best_val = df_hist.loc[df_hist["epoch"] == best_epoch, va_col].values[0]
            ax.scatter([best_epoch], [best_val], color="gold", edgecolor="black", s=90, zorder=5, label=f"Best Epoch ({best_epoch})")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8.5)

    fig.suptitle(
        f"Iris Vision {num_cls}-Class Training Curves ({payload.get('backbone', 'resnet50').upper()})",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Training curves saved to: {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASSES,
    output_path: Path = CONFUSION_MATRIX_PNG,
) -> None:
    """Plot raw counts and normalized recall confusion matrix (supports 4x4 or 5x5)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    sns.set_theme(style="white", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axes[0],
        annot_kws={"size": 11, "weight": "bold"},
    )
    axes[0].set_title(f"Confusion Matrix ({len(class_names)}x{len(class_names)} Sample Counts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("True Class")
    axes[0].tick_params(axis="x", rotation=20)

    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=100.0,
        cbar_kws={"label": "Recall (%)"},
        ax=axes[1],
        annot_kws={"size": 11, "weight": "bold"},
    )
    axes[1].set_title("Confusion Matrix (Row-Normalized Recall %)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("True Class")
    axes[1].tick_params(axis="x", rotation=20)

    acc = accuracy_score(y_true, y_pred)
    fig.suptitle(
        f"Iris Vision {len(class_names)}-Class Test Evaluation - Accuracy: {acc * 100:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Confusion matrix saved to: {output_path}")


def generate_gradcam_visualization(
    model: nn.Module,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    device: torch.device,
    class_names: List[str] = CLASSES,
    output_path: Path = GRADCAM_SAMPLES_PNG,
) -> None:
    """Generate Grad-CAM visual grid across all active classes on test samples."""
    target_layer = model.get_target_layer_for_cam()
    cam_extractor = GradCAM(model, target_layer)
    eval_transform = get_default_transforms(is_train=False)

    selected_indices: list[int] = []
    for cls_idx in range(len(class_names)):
        correct_mask = (y_true == cls_idx) & (y_pred == cls_idx)
        candidate_indices = np.where(correct_mask)[0]
        if len(candidate_indices) > 0:
            best_local = candidate_indices[np.argmax(y_probs[candidate_indices, cls_idx])]
            selected_indices.append(int(best_local))
        else:
            fallback = np.where(y_true == cls_idx)[0]
            if len(fallback) > 0:
                selected_indices.append(int(fallback[0]))

    fig, axes = plt.subplots(
        len(selected_indices), 4, figsize=(16, 3.8 * len(selected_indices)), constrained_layout=True
    )
    if len(selected_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    from PIL import Image

    for row_i, sample_idx in enumerate(selected_indices):
        row = test_df.iloc[sample_idx]
        filepath = row["filepath"]
        true_class = row["class_name"]

        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        orig_rgb = cv2.cvtColor(cv2.resize(img_bgr, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        prep_rgb = preprocess_image(img_bgr, class_name=None, target_size=IMAGE_SIZE, return_rgb=True)

        input_tensor = eval_transform(Image.fromarray(prep_rgb)).unsqueeze(0).to(device)
        cam_map, pred_idx, pred_conf = cam_extractor.generate(input_tensor)
        pred_class = class_names[pred_idx]

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = np.clip(
            0.58 * prep_rgb.astype(np.float32) + 0.42 * heatmap_rgb.astype(np.float32), 0, 255
        ).astype(np.uint8)

        panels = [
            (orig_rgb, f"Original\nTrue: {true_class}"),
            (prep_rgb, f"Preprocessed (ROI+CLAHE)\n{row['filename'][:26]}"),
            (heatmap_rgb, "Grad-CAM Activation\nTarget Layer: features[-1]"),
            (overlay_rgb, f"Overlay Prediction\nPred: {pred_class} ({pred_conf * 100:.1f}%)"),
        ]

        for col_j, (img_vis, title) in enumerate(panels):
            ax = axes[row_i, col_j]
            ax.imshow(img_vis)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

    cam_extractor.close()
    fig.suptitle(
        f"Iris Vision ({len(class_names)}-Class) - Grad-CAM Model Interpretability",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Grad-CAM visualization saved to: {output_path}")


@torch.no_grad()
def compute_ood_and_combined_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute OOD False Acceptance Rate (FAR) for Layer 2 standalone and combined Layer 1 + Layer 2."""
    if OOD_CLASS_NAME not in class_names:
        return {}

    ood_idx = class_names.index(OOD_CLASS_NAME)
    ood_mask = y_true == ood_idx
    eye_mask = ~ood_mask
    n_ood = int(ood_mask.sum())
    n_eye = int(eye_mask.sum())

    # 1. Layer 2 standalone (Argmax)
    l2_argmax_false_accept = int((y_pred[ood_mask] != ood_idx).sum())
    l2_argmax_far = (l2_argmax_false_accept / max(1, n_ood)) * 100.0

    # 2. Layer 2 with production threshold (reject if argmax == non_fundus OR P(non_fundus) > 0.40)
    p_non_fundus = y_probs[:, ood_idx]
    l2_rejected = (y_pred == ood_idx) | (p_non_fundus > L2_NON_FUNDUS_THRESHOLD)
    l2_thresh_false_accept = int((~l2_rejected[ood_mask]).sum())
    l2_thresh_far = (l2_thresh_false_accept / max(1, n_ood)) * 100.0
    l2_eye_pass_rate = float((~l2_rejected[eye_mask]).mean()) * 100.0

    # In-domain 4-class accuracy (on real eye test images, renormalizing over the 4 medical classes)
    med_indices = [i for i, c in enumerate(class_names) if c != OOD_CLASS_NAME]
    med_probs = y_probs[:, med_indices]
    med_pred_local = np.argmax(med_probs, axis=1)
    med_pred_global = np.array([med_indices[i] for i in med_pred_local], dtype=np.int64)
    eye_4class_acc = float(accuracy_score(y_true[eye_mask], med_pred_global[eye_mask])) * 100.0

    metrics = {
        "n_ood_test": n_ood,
        "n_eye_test": n_eye,
        "l2_argmax_far_pct": l2_argmax_far,
        "l2_threshold_far_pct": l2_thresh_far,
        "l2_ood_rejection_rate_pct": 100.0 - l2_thresh_far,
        "l2_eye_pass_rate_pct": l2_eye_pass_rate,
        "ocular_4class_subset_acc_pct": eye_4class_acc,
    }

    # 3. Combined Layer 1 (Gatekeeper) + Layer 2 (5-Class) if gatekeeper checkpoint exists
    if GATEKEEPER_BEST_PATH.exists():
        gk_model, _ = load_trained_model(GATEKEEPER_BEST_PATH, GATEKEEPER_CONFIG_JSON, device=device)
        gk_eye_probs_list: list[np.ndarray] = []
        amp_enabled = (device.type == "cuda") and USE_AMP
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                gk_logits = gk_model(images)
                gk_probs = F.softmax(gk_logits, dim=1)
            gk_eye_probs_list.append(gk_probs[:, 1].cpu().numpy())

        p_valid_eye = np.concatenate(gk_eye_probs_list, axis=0)
        l1_rejected = p_valid_eye < GATEKEEPER_EYE_THRESHOLD
        combined_rejected = l1_rejected | l2_rejected

        comb_false_accept = int((~combined_rejected[ood_mask]).sum())
        comb_far = (comb_false_accept / max(1, n_ood)) * 100.0
        comb_ood_rejection = 100.0 - comb_far
        comb_eye_pass = float((~combined_rejected[eye_mask]).mean()) * 100.0

        metrics["l1_gatekeeper_ood_rejection_pct"] = float(l1_rejected[ood_mask].mean()) * 100.0
        metrics["l1_gatekeeper_eye_pass_pct"] = float((~l1_rejected[eye_mask]).mean()) * 100.0
        metrics["combined_ood_far_pct"] = comb_far
        metrics["combined_ood_rejection_rate_pct"] = comb_ood_rejection
        metrics["combined_eye_pass_rate_pct"] = comb_eye_pass

    return metrics


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    config_path: Path = MODEL_CONFIG_JSON,
) -> Dict[str, float]:
    """Evaluate best saved model (5-class or 4-class) on test set and produce all artifacts."""
    ensure_output_dir()
    device = get_device()

    if checkpoint_path is None:
        checkpoint_path = MODEL_5CLASS_PATH if MODEL_5CLASS_PATH.exists() else MODEL_BEST_PATH

    model, config = load_trained_model(checkpoint_path, config_path, device=device)
    num_classes = int(config.get("num_classes", NUM_CLASSES))
    active_classes: List[str] = list(config.get("classes", CLASSES_5 if num_classes == 5 else CLASSES))

    if num_classes == 5:
        _, _, test_df = get_5class_split_dataframes(
            dataset_dir=DATASET_DIR,
            balance_train=False,
        )
        cm_out = CONFUSION_MATRIX_5CLASS_PNG
        curves_in = TRAINING_HISTORY_5CLASS_JSON if TRAINING_HISTORY_5CLASS_JSON.exists() else TRAINING_HISTORY_JSON
        curves_out = TRAINING_CURVES_5CLASS_PNG
        gradcam_out = GRADCAM_5CLASS_PNG
    else:
        _, _, test_df = get_split_dataframes(
            dataset_dir=DATASET_DIR,
            use_existing_splits=True,
            clean_conflicting_duplicates=True,
            balance_train=False,
        )
        cm_out = CONFUSION_MATRIX_PNG
        curves_in = TRAINING_HISTORY_JSON
        curves_out = TRAINING_CURVES_PNG
        gradcam_out = GRADCAM_SAMPLES_PNG

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
        pin_memory=PIN_MEMORY and (device.type == "cuda"),
    )

    y_true, y_pred, y_probs = evaluate_test_set(model, test_loader, device=device, use_amp=USE_AMP)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    print("\n" + "=" * 76)
    print(
        f"IRIS VISION - {num_classes}-CLASS TEST SET EVALUATION REPORT "
        f"({config.get('backbone', BACKBONE_NAME).upper()})"
    )
    print("=" * 76)
    print(f"Checkpoint         : {checkpoint_path}")
    print(f"Test Samples       : {len(test_df)}")
    print(f"Test Accuracy      : {acc * 100:.2f}% (Target >= 90.00%)")
    print(f"Macro Precision    : {macro_prec * 100:.2f}%")
    print(f"Macro Recall       : {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score     : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score  : {weighted_f1 * 100:.2f}%")
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=active_classes, digits=4, zero_division=0))

    ood_metrics = {}
    if num_classes == 5:
        ood_metrics = compute_ood_and_combined_metrics(
            y_true, y_pred, y_probs, active_classes, test_loader, device
        )
        print("--- OOD Rejection & Defense-in-Depth Metrics ---")
        print(f"OOD Test Samples (`non_fundus`)          : {ood_metrics['n_ood_test']}")
        print(f"Ocular Test Samples (4 Eye Classes)      : {ood_metrics['n_eye_test']}")
        print(f"Layer-2 Standalone OOD FAR (Argmax)      : {ood_metrics['l2_argmax_far_pct']:.2f}%")
        print(
            f"Layer-2 OOD FAR (P(non_fundus) > {L2_NON_FUNDUS_THRESHOLD:.2f})    : "
            f"{ood_metrics['l2_threshold_far_pct']:.2f}% "
            f"(Rejection Rate: {ood_metrics['l2_ood_rejection_rate_pct']:.2f}%)"
        )
        print(f"Ocular 4-Class Renormalized Test Accuracy: {ood_metrics['ocular_4class_subset_acc_pct']:.2f}%")
        if "combined_ood_far_pct" in ood_metrics:
            print(
                f"Combined 2-Layer (Gatekeeper + 5-Class)  : "
                f"OOD FAR = {ood_metrics['combined_ood_far_pct']:.2f}% | "
                f"OOD Rejection = {ood_metrics['combined_ood_rejection_rate_pct']:.2f}% | "
                f"Valid Eye Pass Rate = {ood_metrics['combined_eye_pass_rate_pct']:.2f}%"
            )
    print("=" * 76)

    plot_training_curves(curves_in, curves_out)
    plot_confusion_matrix(y_true, y_pred, active_classes, cm_out)
    if cm_out != CONFUSION_MATRIX_PNG:
        plot_confusion_matrix(y_true, y_pred, active_classes, CONFUSION_MATRIX_PNG)
    generate_gradcam_visualization(
        model, test_df, y_true, y_pred, y_probs, device, active_classes, gradcam_out
    )

    result = {
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }
    result.update(ood_metrics)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Iris Vision model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=str(MODEL_CONFIG_JSON), help="Path to model_config.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    run_evaluation(checkpoint_path=ckpt_path, config_path=Path(args.config))
