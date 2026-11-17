"""Model evaluation, training curve visualization, confusion matrix, and Grad-CAM for Iris Vision."""

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

# Allow running directly as `python src/evaluate.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BACKBONE_NAME,
    BATCH_SIZE,
    CLASSES,
    CONFUSION_MATRIX_PNG,
    DATASET_DIR,
    GRADCAM_SAMPLES_PNG,
    IMAGE_SIZE,
    MODEL_BEST_PATH,
    MODEL_CONFIG_JSON,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAINING_CURVES_PNG,
    TRAINING_HISTORY_JSON,
    USE_AMP,
    ensure_output_dir,
    get_device,
)
from src.dataset import (
    EyeDiseaseDataset,
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
        # Register gradient hook directly on the activation tensor to avoid
        # module-level full backward hook conflicts with inplace activations.
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

        # Global Average Pooling over spatial dimensions (H, W) of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)   # [H, W]
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

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

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

        # Phase 1 -> Phase 2 boundary
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
        f"Iris Vision Training Curves ({payload.get('backbone', 'mobilenet_v2').upper()})",
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
    """Plot raw counts and normalized recall confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    sns.set_theme(style="white", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

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
    axes[0].set_title("Confusion Matrix (Sample Counts)", fontsize=12, fontweight="bold")
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
    fig.suptitle(f"Iris Vision Test Evaluation - Accuracy: {acc * 100:.2f}%", fontsize=14, fontweight="bold")
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
    output_path: Path = GRADCAM_SAMPLES_PNG,
) -> None:
    """Generate Grad-CAM visual grid across all 4 classes on test samples."""
    target_layer = model.get_target_layer_for_cam()
    cam_extractor = GradCAM(model, target_layer)
    eval_transform = get_default_transforms(is_train=False)

    # Pick one representative (highest-confidence correct) test sample per class
    selected_indices: list[int] = []
    for cls_idx in range(NUM_CLASSES):
        correct_mask = (y_true == cls_idx) & (y_pred == cls_idx)
        candidate_indices = np.where(correct_mask)[0]
        if len(candidate_indices) > 0:
            best_local = candidate_indices[np.argmax(y_probs[candidate_indices, cls_idx])]
            selected_indices.append(int(best_local))
        else:
            fallback = np.where(y_true == cls_idx)[0]
            if len(fallback) > 0:
                selected_indices.append(int(fallback[0]))

    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(16, 4 * len(selected_indices)), constrained_layout=True)
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
        pred_class = CLASSES[pred_idx]

        # Colorize heatmap (JET) and blend onto preprocessed RGB
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = np.clip(0.58 * prep_rgb.astype(np.float32) + 0.42 * heatmap_rgb.astype(np.float32), 0, 255).astype(np.uint8)

        panels = [
            (orig_rgb, f"Original\nTrue: {true_class}"),
            (prep_rgb, f"Preprocessed (ROI+CLAHE)\n{row['filename']}"),
            (heatmap_rgb, f"Grad-CAM Activation\nTarget Layer: features[-1]"),
            (overlay_rgb, f"Overlay Prediction\nPred: {pred_class} ({pred_conf * 100:.1f}%)"),
        ]

        for col_j, (img_vis, title) in enumerate(panels):
            ax = axes[row_i, col_j]
            ax.imshow(img_vis)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

    cam_extractor.close()
    fig.suptitle("Iris Vision - Grad-CAM Model Interpretability (Test Set)", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Grad-CAM visualization saved to: {output_path}")


def run_evaluation() -> Dict[str, float]:
    """Evaluate best saved model on test set and produce all evaluation artifacts."""
    ensure_output_dir()
    device = get_device()
    model, config = load_trained_model(MODEL_BEST_PATH, MODEL_CONFIG_JSON, device=device)

    _, _, test_df = get_split_dataframes(
        dataset_dir=DATASET_DIR,
        use_existing_splits=True,
        clean_conflicting_duplicates=True,
        balance_train=False,
    )

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

    print("\n" + "=" * 72)
    print(f"IRIS VISION - TEST SET EVALUATION REPORT ({config.get('backbone', BACKBONE_NAME).upper()})")
    print("=" * 72)
    print(f"Test Samples       : {len(test_df)}")
    print(f"Test Accuracy      : {acc * 100:.2f}% (Target >= 90.00%)")
    print(f"Macro Precision    : {macro_prec * 100:.2f}%")
    print(f"Macro Recall       : {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score     : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score  : {weighted_f1 * 100:.2f}%")
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))
    print("=" * 72)

    plot_training_curves(TRAINING_HISTORY_JSON, TRAINING_CURVES_PNG)
    plot_confusion_matrix(y_true, y_pred, CLASSES, CONFUSION_MATRIX_PNG)
    generate_gradcam_visualization(model, test_df, y_true, y_pred, y_probs, device, GRADCAM_SAMPLES_PNG)

    return {
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }


if __name__ == "__main__":
    run_evaluation()
