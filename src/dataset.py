"""PyTorch Dataset, stratified splitting, and class balancing for Iris Vision.

Designed for full Pandas 3.0+ compatibility using explicit `pd.concat` per group
and clean stratified splitting with zero train-to-val/test leakage.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
from typing import Callable, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    AUDIT_REPORT_CSV,
    BATCH_SIZE,
    CLASS_TO_IDX,
    CLASSES,
    CLEAN_MANIFEST_CSV,
    DATASET_DIR,
    FOLDER_TO_CLASS,
    IGNORED_FOLDERS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_TRAIN_SAMPLES_PER_CLASS,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    SPLITS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    VALID_EXTENSIONS,
    ensure_output_dir,
)
from src.preprocess import preprocess_image


def scan_dataset_dataframe(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Scan dataset_dir/{split}/{class_folder} into a Pandas DataFrame.

    Normalizes folder names via `FOLDER_TO_CLASS` and skips `IGNORED_FOLDERS` (e.g. 'other').
    """
    records: list[dict] = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for sub_dir in sorted(split_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            folder_key = sub_dir.name.lower().strip()
            if folder_key in IGNORED_FOLDERS or folder_key not in FOLDER_TO_CLASS:
                continue
            class_name = FOLDER_TO_CLASS[folder_key]
            for file_path in sorted(sub_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                    fname = file_path.name
                    records.append(
                        {
                            "filepath": str(file_path.resolve()),
                            "filename": fname,
                            "class_name": class_name,
                            "label_idx": CLASS_TO_IDX[class_name],
                            "orig_split": split,
                            "is_augmented": fname.lower().startswith("aug_"),
                        }
                    )

    return pd.DataFrame(records)


def balance_classes_df(
    df: pd.DataFrame,
    strategy: str = "oversample",
    target_count: Optional[int] = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Balance class frequencies using explicit `pd.concat` per group (Pandas 3.0+ safe).

    Args:
        df: Input DataFrame containing a `class_name` column.
        strategy: 'oversample' (upsample minority classes to max count),
                  'downsample' (downsample majority classes to min count),
                  or 'target' (sample all classes to `target_count`).
        target_count: Desired number of samples per class when strategy='target'.
        seed: Random seed for reproducible sampling.
    """
    if df.empty:
        return df.copy()

    class_counts = df["class_name"].value_counts()
    if strategy == "oversample":
        n_target = int(class_counts.max())
    elif strategy == "downsample":
        n_target = int(class_counts.min())
    elif strategy == "target":
        if target_count is None or target_count <= 0:
            raise ValueError("`target_count` must be a positive integer when strategy='target'.")
        n_target = int(target_count)
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    balanced_groups: list[pd.DataFrame] = []
    for _, group_df in df.groupby("class_name", sort=True):
        current_n = len(group_df)
        if current_n == n_target:
            sampled = group_df.copy()
        elif current_n < n_target:
            sampled = group_df.sample(n=n_target, replace=True, random_state=seed)
        else:
            sampled = group_df.sample(n=n_target, replace=False, random_state=seed)
        balanced_groups.append(sampled)

    balanced_df = pd.concat(balanced_groups, ignore_index=True)
    # Shuffle rows deterministically
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    balance_train: bool = True,
    balance_strategy: str = "oversample",
    exclude_pre_augmented: bool = False,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test DataFrames with Pandas 3.0+ safe `pd.concat`.

    Balancing is applied strictly to the training split after splitting so that
    validation and test distributions remain free of synthetic duplicates.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    work_df = df.copy()
    if exclude_pre_augmented and "is_augmented" in work_df.columns:
        work_df = work_df[~work_df["is_augmented"]].reset_index(drop=True)

    train_groups: list[pd.DataFrame] = []
    val_groups: list[pd.DataFrame] = []
    test_groups: list[pd.DataFrame] = []

    for _, group_df in work_df.groupby("class_name", sort=True):
        grp = group_df.reset_index(drop=True)
        train_grp, temp_grp = train_test_split(
            grp,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
        )
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_grp, test_grp = train_test_split(
            temp_grp,
            train_size=relative_val_ratio,
            random_state=seed,
            shuffle=True,
        )
        train_groups.append(train_grp)
        val_groups.append(val_grp)
        test_groups.append(test_grp)

    train_df = pd.concat(train_groups, ignore_index=True)
    val_df = pd.concat(val_groups, ignore_index=True)
    test_df = pd.concat(test_groups, ignore_index=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    if balance_train:
        train_df = balance_classes_df(train_df, strategy=balance_strategy, seed=seed)

    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def get_default_transforms(is_train: bool = False) -> transforms.Compose:
    """Return torchvision transform pipeline for training or evaluation."""
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class EyeDiseaseDataset(Dataset):
    """PyTorch Dataset for 4-class ocular disease classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        use_preprocess: bool = True,
        pass_class_name_to_roi: bool = False,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        preload_in_memory: bool = False,
        preload_workers: int = 8,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_preprocess = use_preprocess
        self.pass_class_name_to_roi = pass_class_name_to_roi
        self.target_size = target_size
        self._cached_rgb: Optional[list[np.ndarray]] = None

        if preload_in_memory and len(self.df) > 0:
            self._cached_rgb = self._preload_images(max_workers=preload_workers)

    def _load_single_rgb(self, args: Tuple[str, Optional[str]]) -> np.ndarray:
        filepath, class_name = args
        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Failed to read image at path: {filepath}")

        if self.use_preprocess:
            return preprocess_image(
                img_bgr,
                class_name=class_name,
                target_size=self.target_size,
                return_rgb=True,
            )
        resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _preload_images(self, max_workers: int = 8) -> list[np.ndarray]:
        tasks = [
            (
                row["filepath"],
                row["class_name"] if self.pass_class_name_to_roi else None,
            )
            for _, row in self.df.iterrows()
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            cached = list(executor.map(self._load_single_rgb, tasks))
        print(
            f"[PRELOAD] Cached {len(cached)} preprocessed images "
            f"({self.target_size[0]}x{self.target_size[1]} RGB) in RAM."
        )
        return cached

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label_idx = int(row["label_idx"])

        if self._cached_rgb is not None:
            img_rgb = self._cached_rgb[idx]
        else:
            class_name = row["class_name"] if self.pass_class_name_to_roi else None
            img_rgb = self._load_single_rgb((row["filepath"], class_name))

        pil_img = Image.fromarray(img_rgb)

        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = get_default_transforms(is_train=False)(pil_img)

        return tensor_img, label_idx


def _compute_file_md5(filepath: str, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum for an image file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_base_filename(filename: str) -> str:
    """Strip offline augmentation prefixes and suffixes to get canonical base name."""
    name = re.sub(r"^aug_[a-zA-Z0-9]+_\d+_", "", filename)
    name = re.sub(r"^aug_\d+_", "", name)
    name = re.sub(r"_0_\d+(\.[a-zA-Z0-9]+)$", r"\1", name)
    return name


def clean_conflicting_duplicates_df(
    df: pd.DataFrame,
    audit_csv_path: Path = AUDIT_REPORT_CSV,
    save_manifest_path: Optional[Path] = CLEAN_MANIFEST_CSV,
) -> pd.DataFrame:
    """O(N) scalable duplicate & split-leakage cleaning for large ocular datasets.

    1. Base-Name Split Leakage Pruning:
       - Removes any `train` row (original or `aug_*`) whose `(class_name, base_name)`
         appears in `val` or `test`, keeping `val` and `test` 100% intact.
    2. Parallel MD5 Checksum Deduplication:
       - Computes MD5 in parallel (`ThreadPoolExecutor`) for non-augmented files
         (and all files when `len <= 15000`).
       - Drops all occurrences of any MD5 shared across >1 distinct class label
         plus any `aug_*` copies derived from their base names.
       - Deduplicates identical-label MD5 copies prioritizing `test` > `val` > `train`.
    """
    work_df = df.copy().reset_index(drop=True)
    initial_count = len(work_df)
    work_df["base_name"] = work_df["filename"].apply(_extract_base_filename)

    # Step 1: Prune train rows (and aug_ derivatives) leaking into val/test by base_name
    eval_mask = work_df["orig_split"].isin(["val", "test"])
    eval_keys = set(
        zip(
            work_df.loc[eval_mask, "class_name"],
            work_df.loc[eval_mask, "base_name"],
        )
    )
    train_leak_mask = (work_df["orig_split"] == "train") & pd.Series(
        [(c, b) in eval_keys for c, b in zip(work_df["class_name"], work_df["base_name"])],
        index=work_df.index,
    )
    base_leak_removed = int(train_leak_mask.sum())
    work_df = work_df[~train_leak_mask].reset_index(drop=True)

    # Step 2: Compute MD5 hashes (reusing audit CSV if available, else parallel hashing)
    if "md5_hash" not in work_df.columns:
        if audit_csv_path.exists():
            audit_df = pd.read_csv(audit_csv_path, usecols=["filepath", "md5_hash"])
            hash_map = dict(zip(audit_df["filepath"], audit_df["md5_hash"]))
            work_df["md5_hash"] = work_df["filepath"].map(hash_map)
        else:
            work_df["md5_hash"] = pd.Series(dtype="object")

        # Hash non-augmented files (or all files when dataset <= 15k) in parallel
        if len(work_df) > 15000:
            to_hash_mask = work_df["md5_hash"].isna() & (~work_df["is_augmented"])
            aug_missing_mask = work_df["md5_hash"].isna() & work_df["is_augmented"]
            if aug_missing_mask.any():
                work_df.loc[aug_missing_mask, "md5_hash"] = [
                    f"aug_unique_{i}" for i in work_df.index[aug_missing_mask]
                ]
        else:
            to_hash_mask = work_df["md5_hash"].isna()

        if to_hash_mask.any():
            paths_to_hash = work_df.loc[to_hash_mask, "filepath"].tolist()
            with ThreadPoolExecutor(max_workers=8) as executor:
                hashes = list(executor.map(_compute_file_md5, paths_to_hash))
            work_df.loc[to_hash_mask, "md5_hash"] = hashes

    # Prioritize test > val > train before deduplication
    split_priority = {"test": 0, "val": 1, "train": 2}
    work_df["_split_prio"] = work_df["orig_split"].map(split_priority).fillna(3)
    work_df = work_df.sort_values(["_split_prio", "is_augmented"]).reset_index(drop=True)

    # Drop MD5 conflicts across >1 distinct class
    class_nunique_per_hash = work_df.groupby("md5_hash")["class_name"].nunique()
    conflicting_hashes = set(class_nunique_per_hash[class_nunique_per_hash > 1].index)
    md5_conflict_removed = int(work_df["md5_hash"].isin(conflicting_hashes).sum())

    conflicting_base_names = set(
        work_df.loc[work_df["md5_hash"].isin(conflicting_hashes), "base_name"]
    )
    cleaned_df = work_df[~work_df["md5_hash"].isin(conflicting_hashes)].copy()

    # Remove any aug_ images derived from conflicting base names
    aug_conflict_mask = cleaned_df["is_augmented"] & cleaned_df["base_name"].isin(conflicting_base_names)
    aug_conflict_removed = int(aug_conflict_mask.sum())
    cleaned_df = cleaned_df[~aug_conflict_mask].copy()

    # Deduplicate same-label exact MD5 copies keeping test/val priority
    before_dedup = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["md5_hash"], keep="first").reset_index(drop=True)
    md5_same_removed = before_dedup - len(cleaned_df)
    cleaned_df = cleaned_df.drop(columns=["_split_prio"], errors="ignore")

    print(
        f"[MANIFEST CLEAN] Initial: {initial_count} | "
        f"Train-to-Val/Test base leaks removed: {base_leak_removed} | "
        f"MD5 conflicts: {md5_conflict_removed} | Aug-derived conflicts: {aug_conflict_removed} | "
        f"MD5 same-label dedup: {md5_same_removed} | Clean total: {len(cleaned_df)}"
    )

    if save_manifest_path is not None:
        ensure_output_dir()
        cleaned_df.to_csv(save_manifest_path, index=False)

    return cleaned_df


def sample_balanced_train_df(
    train_df: pd.DataFrame,
    max_per_class: int = MAX_TRAIN_SAMPLES_PER_CLASS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Balance train DataFrame up to `max_per_class`, prioritizing 100% original images first."""
    if train_df.empty or max_per_class <= 0:
        return train_df.copy()

    balanced_groups: list[pd.DataFrame] = []
    for _, group_df in train_df.groupby("class_name", sort=True):
        if len(group_df) <= max_per_class:
            balanced_groups.append(group_df.copy())
            continue

        orig_df = group_df[~group_df["is_augmented"]]
        aug_df = group_df[group_df["is_augmented"]]

        if len(orig_df) >= max_per_class:
            sampled = orig_df.sample(n=max_per_class, replace=False, random_state=seed)
        else:
            needed_aug = max_per_class - len(orig_df)
            sampled_aug = aug_df.sample(
                n=min(needed_aug, len(aug_df)), replace=False, random_state=seed
            )
            sampled = pd.concat([orig_df, sampled_aug], ignore_index=True)

        balanced_groups.append(sampled)

    balanced_df = pd.concat(balanced_groups, ignore_index=True)
    return balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Compute balanced inverse-frequency weights ordered by CLASSES index."""
    counts = df["label_idx"].value_counts().to_dict()
    total = float(len(df))
    weights = []
    for idx in range(NUM_CLASSES):
        cnt = float(counts.get(idx, 1))
        weights.append(total / (NUM_CLASSES * cnt))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize mean to 1.0 for loss magnitude stability
    return weight_tensor / weight_tensor.mean()


def get_split_dataframes(
    dataset_dir: Path = DATASET_DIR,
    use_existing_splits: bool = True,
    clean_conflicting_duplicates: bool = True,
    balance_train: bool = False,
    balance_strategy: str = "oversample",
    max_train_per_class: int = MAX_TRAIN_SAMPLES_PER_CLASS,
    force_reclean: bool = False,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cleaned (train_df, val_df, test_df) splits consistently."""
    use_cached = (
        clean_conflicting_duplicates
        and CLEAN_MANIFEST_CSV.exists()
        and not force_reclean
    )
    if use_cached:
        master_df = pd.read_csv(CLEAN_MANIFEST_CSV)
        expected_prefix = str(dataset_dir.resolve())
        if master_df.empty or not str(master_df.iloc[0]["filepath"]).startswith(expected_prefix):
            use_cached = False

    if not use_cached:
        master_df = scan_dataset_dataframe(dataset_dir)
        if clean_conflicting_duplicates:
            master_df = clean_conflicting_duplicates_df(master_df)

    if use_existing_splits:
        train_df = master_df[master_df["orig_split"] == "train"].reset_index(drop=True)
        val_df = master_df[master_df["orig_split"] == "val"].reset_index(drop=True)
        test_df = master_df[master_df["orig_split"] == "test"].reset_index(drop=True)
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        if max_train_per_class > 0:
            train_df = sample_balanced_train_df(
                train_df, max_per_class=max_train_per_class, seed=seed
            )
        if balance_train:
            train_df = balance_classes_df(train_df, strategy=balance_strategy, seed=seed)
    else:
        train_df, val_df, test_df = create_stratified_splits(
            master_df,
            balance_train=balance_train,
            balance_strategy=balance_strategy,
            seed=seed,
        )

    return train_df, val_df, test_df


def get_dataloaders(
    dataset_dir: Path = DATASET_DIR,
    use_existing_splits: bool = True,
    clean_conflicting_duplicates: bool = True,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    balance_train: bool = False,
    balance_strategy: str = "oversample",
    max_train_per_class: int = MAX_TRAIN_SAMPLES_PER_CLASS,
    preload_in_memory: bool = False,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train, val, and test PyTorch DataLoaders."""
    train_df, val_df, test_df = get_split_dataframes(
        dataset_dir=dataset_dir,
        use_existing_splits=use_existing_splits,
        clean_conflicting_duplicates=clean_conflicting_duplicates,
        balance_train=balance_train,
        balance_strategy=balance_strategy,
        max_train_per_class=max_train_per_class,
        seed=seed,
    )

    train_dataset = EyeDiseaseDataset(
        train_df,
        transform=get_default_transforms(is_train=True),
        use_preprocess=True,
        preload_in_memory=preload_in_memory,
    )
    val_dataset = EyeDiseaseDataset(
        val_df,
        transform=get_default_transforms(is_train=False),
        use_preprocess=True,
        preload_in_memory=preload_in_memory,
    )
    test_dataset = EyeDiseaseDataset(
        test_df,
        transform=get_default_transforms(is_train=False),
        use_preprocess=True,
        preload_in_memory=preload_in_memory,
    )

    effective_workers = 0 if preload_in_memory else num_workers
    persistent = effective_workers > 0
    pin = PIN_MEMORY and torch.cuda.is_available()

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, CLASSES

