"""Central configuration for the Iris Vision ocular disease classification project."""

from pathlib import Path
import torch

# ==============================================================================
# Directory Paths
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
OLD_DATASET_DIR: Path = BASE_DIR / "dataset2_balanced"
DATASET_DIR: Path = BASE_DIR / "Retinal Fundus Images"
SRC_DIR: Path = BASE_DIR / "src"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
MODELS_DIR: Path = BASE_DIR / "models"

# Audit & Manifest Paths
AUDIT_REPORT_CSV: Path = OUTPUTS_DIR / "dataset_audit_report.csv"
CLEAN_MANIFEST_CSV: Path = OUTPUTS_DIR / "clean_dataset_manifest.csv"
EDA_SUMMARY_PNG: Path = OUTPUTS_DIR / "eda_summary.png"
STRATIFIED_MANIFEST_CSV: Path = OUTPUTS_DIR / "stratified_manifest.csv"

# Model Checkpoint & Evaluation Output Paths
MODEL_BEST_PATH: Path = MODELS_DIR / "ai_sentinel_best.pth"
MODEL_CONFIG_JSON: Path = MODELS_DIR / "model_config.json"
TRAINING_HISTORY_JSON: Path = OUTPUTS_DIR / "training_history.json"
TRAINING_CURVES_PNG: Path = OUTPUTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PNG: Path = OUTPUTS_DIR / "confusion_matrix.png"
GRADCAM_SAMPLES_PNG: Path = OUTPUTS_DIR / "gradcam_samples.png"

# ==============================================================================
# Dataset Classes & Mappings
# ==============================================================================
SPLITS: list[str] = ["train", "val", "test"]
CLASSES: list[str] = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
NUM_CLASSES: int = len(CLASSES)

CLASS_TO_IDX: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

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
# Phase 3 Inference & API Parameters
# ==============================================================================
CONFIDENCE_THRESHOLD: float = 0.80  # 80% threshold for confident diagnosis


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

