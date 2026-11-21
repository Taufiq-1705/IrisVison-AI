"""Central configuration for the Iris Vision ocular disease classification project."""

from pathlib import Path
import torch

# ==============================================================================
# Directory Paths
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATASETS_DIR: Path = BASE_DIR / "DATASETS"
RETINAL_FUNDUS_DIR: Path = DATASETS_DIR / "Retinal Fundus Images"
DATASET_DIR: Path = DATASETS_DIR / "dataset2_balanced(Updated)"
OLD_DATASET_DIR: Path = DATASET_DIR
RANDOM_IMAGE_OOD_DIR: Path = DATASETS_DIR / "class-random-image"
SRC_DIR: Path = BASE_DIR / "src"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
MODELS_DIR: Path = BASE_DIR / "models"

# Audit & Manifest Paths
AUDIT_REPORT_CSV: Path = OUTPUTS_DIR / "dataset_audit_report.csv"
CLEAN_MANIFEST_CSV: Path = OUTPUTS_DIR / "clean_dataset_manifest.csv"
OOD_MANIFEST_CSV: Path = OUTPUTS_DIR / "ood_manifest.csv"
EDA_SUMMARY_PNG: Path = OUTPUTS_DIR / "eda_summary.png"
STRATIFIED_MANIFEST_CSV: Path = OUTPUTS_DIR / "stratified_manifest.csv"

# Model Checkpoint & Evaluation Output Paths
MODEL_BEST_PATH: Path = MODELS_DIR / "ai_sentinel_best.pth"
MODEL_CONFIG_JSON: Path = MODELS_DIR / "model_config.json"
MODEL_5CLASS_PATH: Path = MODELS_DIR / "ai_sentinel_5class_best.pth"
GATEKEEPER_BEST_PATH: Path = MODELS_DIR / "gatekeeper_best.pth"
GATEKEEPER_CONFIG_JSON: Path = MODELS_DIR / "gatekeeper_config.json"

RESNET50_5CLASS_DIR: Path = MODELS_DIR / "resnet50_5class"
RESNET50_5CLASS_PATH: Path = RESNET50_5CLASS_DIR / "ai_sentinel_resnet50_5class.pth"
RESNET50_5CLASS_CONFIG_JSON: Path = RESNET50_5CLASS_DIR / "model_config.json"

MOBILENETV2_5CLASS_DIR: Path = MODELS_DIR / "mobilenetv2_5class"
MOBILENETV2_5CLASS_PATH: Path = MOBILENETV2_5CLASS_DIR / "ai_sentinel_mobilenetv2_5class.pth"
MOBILENETV2_5CLASS_CONFIG_JSON: Path = MOBILENETV2_5CLASS_DIR / "model_config.json"

TRAINING_HISTORY_JSON: Path = OUTPUTS_DIR / "training_history.json"
TRAINING_CURVES_PNG: Path = OUTPUTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PNG: Path = OUTPUTS_DIR / "confusion_matrix.png"
GRADCAM_SAMPLES_PNG: Path = OUTPUTS_DIR / "gradcam_samples.png"

TRAINING_HISTORY_5CLASS_JSON: Path = OUTPUTS_DIR / "training_history_5class.json"
TRAINING_CURVES_5CLASS_PNG: Path = OUTPUTS_DIR / "training_curves_5class.png"
CONFUSION_MATRIX_5CLASS_PNG: Path = OUTPUTS_DIR / "confusion_matrix_5class.png"
GRADCAM_5CLASS_PNG: Path = OUTPUTS_DIR / "gradcam_5class_samples.png"

GATEKEEPER_HISTORY_JSON: Path = OUTPUTS_DIR / "gatekeeper_training_history.json"
GATEKEEPER_CURVES_PNG: Path = OUTPUTS_DIR / "gatekeeper_training_curves.png"
GATEKEEPER_CONFUSION_MATRIX_PNG: Path = OUTPUTS_DIR / "gatekeeper_confusion_matrix.png"

# ==============================================================================
# Dataset Classes & Mappings
# ==============================================================================
SPLITS: list[str] = ["train", "val", "test"]
CLASSES: list[str] = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
CLASSES_4: list[str] = CLASSES
NUM_CLASSES: int = len(CLASSES)

CLASS_TO_IDX: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# 5-Class Defense-in-Depth Definitions (Layer 2)
OOD_CLASS_NAME: str = "non_fundus"
CLASSES_5: list[str] = ["cataract", "diabetic_retinopathy", "glaucoma", "non_fundus", "normal"]
NUM_CLASSES_5: int = len(CLASSES_5)
CLASS_TO_IDX_5: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES_5)}
IDX_TO_CLASS_5: dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASSES_5)}

# 2-Class Gatekeeper Definitions (Layer 1)
GATEKEEPER_CLASSES: list[str] = ["non_eye", "valid_eye"]
GATEKEEPER_CLASS_TO_IDX: dict[str, int] = {"non_eye": 0, "valid_eye": 1}
GATEKEEPER_IDX_TO_CLASS: dict[int, str] = {0: "non_eye", 1: "valid_eye"}

FOLDER_TO_CLASS: dict[str, str] = {
    "cataract": "cataract",
    "diabetic_retinopathy": "diabetic_retinopathy",
    "glaucoma": "glaucoma",
    "normal": "normal",
    "normal_fundus": "normal",
}
IGNORED_FOLDERS: set[str] = {"other"}
MAX_TRAIN_SAMPLES_PER_CLASS: int = 3000

VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==============================================================================
# Preprocessing & Audit Parameters
# ==============================================================================
IMAGE_SIZE: tuple[int, int] = (224, 224)

# LAB CLAHE Parameters
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

# Fundus ROI & Border Analysis
FUNDUS_BORDER_RATIO: float = 0.05
FUNDUS_BORDER_BRIGHTNESS_THRESHOLD: float = 25.0
FUNDUS_ROI_THRESHOLD: int = 12
MIN_ROI_AREA_RATIO: float = 0.20

# Blur Detection
BLUR_LAPLACIAN_THRESHOLD: float = 100.0

# Normalization (ImageNet Standard)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# ==============================================================================
# Hardware & Training Parameters (Windows 11 Native - RTX 3050 6GB)
# ==============================================================================
SEED: int = 42
BATCH_SIZE: int = 32
NUM_WORKERS: int = 2
PIN_MEMORY: bool = True
USE_AMP: bool = True

# Default Stratified Split Ratios (when rebuilding splits from master data)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ==============================================================================
# Phase 2 Model & Training Hyperparameters
# ==============================================================================
BACKBONE_NAME: str = "mobilenet_v2"  # Supported: "mobilenet_v2", "efficientnet_b0"
WARMUP_EPOCHS: int = 4
FINETUNE_EPOCHS: int = 16
WARMUP_LR: float = 1e-3
BACKBONE_LR: float = 1e-4
HEAD_LR: float = 3e-4
WEIGHT_DECAY: float = 1e-4
LABEL_SMOOTHING: float = 0.05
EARLY_STOPPING_PATIENCE: int = 5

# ==============================================================================
# Phase 3 Inference, Validation, Defense-in-Depth OOD, & QA Robustness Parameters
# ==============================================================================
CONFIDENCE_THRESHOLD: float = 0.90        # 90% threshold for confident diagnosis
GATEKEEPER_EYE_THRESHOLD: float = 0.65    # Layer 1 minimum P(valid_eye) to pass gatekeeper
L2_NON_FUNDUS_THRESHOLD: float = 0.40     # Layer 2 max allowed P(non_fundus) before OOD rejection
MIN_IMAGE_DIMENSION: int = 224            # Minimum required width & height in pixels
ALLOWED_API_FORMATS: set[str] = {"JPEG", "PNG"}
ROBUSTNESS_OUTPUT_DIR: Path = OUTPUTS_DIR / "robustness"

# Default OOD split target counts
OOD_TRAIN_TARGET: int = 1150
OOD_VAL_TARGET: int = 115
OOD_TEST_TARGET: int = 200


def get_device() -> torch.device:
    """Return the active PyTorch compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir() -> Path:
    """Create the outputs directory if it does not exist and return it."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def ensure_model_dir() -> Path:
    """Create the models directory if it does not exist and return it."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR

