"""Automated dataset audit and EDA visualization for Iris Vision.

Audits `dataset2_balanced` across train/val/test splits for:
- Corrupt / unreadable images (`cv2.imread`)
- Image dimensions, channels, and aspect ratios
- Class distribution and Imbalance Ratio (IR) per split
- Cross-split & within-split duplicate detection via MD5 hashing
- Blur detection via Variance of Laplacian (`cv2.Laplacian`)
- Edge / border brightness analysis proving cataract vs. fundus framing difference

Saves:
- Detailed per-image audit CSV -> outputs/dataset_audit_report.csv
- EDA plots + outlier sample thumbnails -> outputs/eda_summary.png
"""

import hashlib
import os
from pathlib import Path
import sys
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Windows headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Allow running directly as `python src/audit_dataset.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AUDIT_REPORT_CSV,
    BLUR_LAPLACIAN_THRESHOLD,
    CLASSES,
    DATASET_DIR,
    EDA_SUMMARY_PNG,
    FUNDUS_BORDER_BRIGHTNESS_THRESHOLD,
    FUNDUS_BORDER_RATIO,
    SPLITS,
    VALID_EXTENSIONS,
    ensure_output_dir,
)
from src.preprocess import compute_border_brightness

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_md5(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_center_brightness(img_bgr: np.ndarray, center_ratio: float = 0.5) -> float:
    """Compute mean grayscale luminance in the central region of the image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = gray.shape[:2]
    ch = max(1, int(h * center_ratio))
    cw = max(1, int(w * center_ratio))
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    return float(gray[y1 : y1 + ch, x1 : x1 + cw].mean())


def audit_single_image(filepath: Path, split: str, class_name: str) -> dict:
    """Audit a single image file for integrity, geometry, hash, blur, and border brightness."""
    file_size_kb = round(os.path.getsize(filepath) / 1024.0, 2)
    md5_hash = compute_md5(filepath)

    # Read with IMAGED_UNCHANGED to detect true channel count (1, 3, or 4)
    img_raw = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if img_raw is None or img_raw.size == 0:
        return {
            "split": split,
            "class_name": class_name,
            "filename": filepath.name,
            "filepath": str(filepath.resolve()),
            "is_corrupt": True,
            "width": np.nan,
            "height": np.nan,
            "channels": np.nan,
            "aspect_ratio": np.nan,
            "file_size_kb": file_size_kb,
            "laplacian_var": np.nan,
            "is_blurry": np.nan,
            "border_brightness": np.nan,
            "center_brightness": np.nan,
            "is_augmented": filepath.name.lower().startswith("aug_"),
            "md5_hash": md5_hash,
        }

    h, w = img_raw.shape[:2]
    channels = img_raw.shape[2] if img_raw.ndim == 3 else 1

    # Standardize to BGR for luminance/blur metrics
    if channels == 1:
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img_raw

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    border_brightness = compute_border_brightness(img_bgr, border_ratio=FUNDUS_BORDER_RATIO)
    center_brightness = compute_center_brightness(img_bgr, center_ratio=0.50)

    return {
        "split": split,
        "class_name": class_name,
        "filename": filepath.name,
        "filepath": str(filepath.resolve()),
        "is_corrupt": False,
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "aspect_ratio": round(w / float(h), 4) if h > 0 else np.nan,
        "file_size_kb": file_size_kb,
        "laplacian_var": round(laplacian_var, 4),
        "is_blurry": bool(laplacian_var < BLUR_LAPLACIAN_THRESHOLD),
        "border_brightness": round(border_brightness, 4),
        "center_brightness": round(center_brightness, 4),
        "is_augmented": filepath.name.lower().startswith("aug_"),
        "md5_hash": md5_hash,
    }


def run_dataset_audit(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Walk all splits and classes in dataset_dir and return audit DataFrame."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    file_tasks: list[tuple[Path, str, str]] = []
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for p in sorted(class_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                    file_tasks.append((p, split, class_name))

    print(f"[INFO] Auditing {len(file_tasks)} images in {dataset_dir} ...")
    records = [
        audit_single_image(p, split, cls_name)
        for p, split, cls_name in tqdm(file_tasks, desc="Auditing images", unit="img")
    ]

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid image files discovered during audit.")

    # Duplicate & Cross-Split Duplicate Detection
    hash_counts = df["md5_hash"].value_counts()
    df["duplicate_count"] = df["md5_hash"].map(hash_counts)
    df["is_duplicate"] = df["duplicate_count"] > 1

    split_nunique_per_hash = df.groupby("md5_hash")["split"].nunique()
    df["splits_with_hash"] = df["md5_hash"].map(split_nunique_per_hash)
    df["is_cross_split_duplicate"] = df["splits_with_hash"] > 1

    return df


def print_audit_summary(df: pd.DataFrame) -> None:
    """Print structured statistical summary of the dataset audit."""
    print("\n" + "=" * 72)
    print("IRIS VISION - DATASET AUDIT REPORT SUMMARY")
    print("=" * 72)

    total_imgs = len(df)
    corrupt_imgs = int(df["is_corrupt"].sum())
    print(f"Total Images Scanned : {total_imgs}")
    print(f"Corrupt Images       : {corrupt_imgs}")

    valid_df = df[~df["is_corrupt"]].copy()

    # 1. Distribution & Imbalance Ratio per Split
    print("\n--- 1. Class Distribution & Imbalance Ratio (IR) per Split ---")
    dist_table = pd.crosstab(valid_df["split"], valid_df["class_name"])
    # Reorder rows and columns cleanly
    dist_table = dist_table.reindex(index=[s for s in SPLITS if s in dist_table.index],
                                    columns=[c for c in CLASSES if c in dist_table.columns],
                                    fill_value=0)
    dist_table["Total"] = dist_table.sum(axis=1)
    imbalance_ratios = {}
    for split_name in dist_table.index:
        cls_counts = dist_table.loc[split_name, CLASSES]
        min_c = int(cls_counts.min())
        max_c = int(cls_counts.max())
        ir = round(max_c / min_c, 2) if min_c > 0 else float("inf")
        imbalance_ratios[split_name] = ir
    dist_table["Imbalance_Ratio"] = pd.Series(imbalance_ratios)
    print(dist_table.to_string())

    # 2. Dimensions & Channels
    print("\n--- 2. Image Dimensions & Channels ---")
    print(f"Channels distribution: {valid_df['channels'].value_counts().to_dict()}")
    dim_stats = valid_df.groupby("class_name")[["width", "height", "aspect_ratio"]].agg(["min", "median", "max"])
    print(dim_stats.to_string())

    # 3. Duplicate & Cross-Split Leakage Analysis
    print("\n--- 3. MD5 Duplicate & Cross-Split Leakage Analysis ---")
    dup_hashes = valid_df[valid_df["is_duplicate"]]["md5_hash"].nunique()
    cross_split_rows = valid_df[valid_df["is_cross_split_duplicate"]]
    cross_hashes = cross_split_rows["md5_hash"].nunique()
    print(f"Total duplicate file clusters (same MD5)  : {dup_hashes}")
    print(f"Total images involved in any duplicate    : {int(valid_df['is_duplicate'].sum())}")
    print(f"Cross-split duplicate MD5 clusters        : {cross_hashes}")
    print(f"Total images involved in cross-split leak : {len(cross_split_rows)}")
    if cross_hashes > 0:
        print("Sample cross-split duplicate clusters (up to 5):")
        for h, grp in list(cross_split_rows.groupby("md5_hash"))[:5]:
            locs = [f"{r['split']}/{r['class_name']}/{r['filename']}" for _, r in grp.iterrows()]
            print(f"  MD5 {h[:10]} -> {locs}")

    # 4. Blur Detection
    print(f"\n--- 4. Blur Detection (Laplacian Variance < {BLUR_LAPLACIAN_THRESHOLD}) ---")
    blur_stats = valid_df.groupby("class_name").agg(
        count=("laplacian_var", "count"),
        mean_lap_var=("laplacian_var", "mean"),
        median_lap_var=("laplacian_var", "median"),
        min_lap_var=("laplacian_var", "min"),
        blurry_count=("is_blurry", "sum"),
        aug_count=("is_augmented", "sum"),
    )
    blur_stats["blurry_pct"] = (blur_stats["blurry_count"] / blur_stats["count"] * 100).round(2)
    print(blur_stats.round(2).to_string())

    # 5. Border Brightness Analysis (Cataract vs Fundus)
    print(f"\n--- 5. Border Brightness (Outer {int(FUNDUS_BORDER_RATIO*100)}% Margin, Threshold={FUNDUS_BORDER_BRIGHTNESS_THRESHOLD}) ---")
    border_stats = valid_df.groupby("class_name").agg(
        mean_border=("border_brightness", "mean"),
        median_border=("border_brightness", "median"),
        std_border=("border_brightness", "std"),
        p5_border=("border_brightness", lambda x: np.percentile(x, 5)),
        p95_border=("border_brightness", lambda x: np.percentile(x, 95)),
        pct_above_thresh=("border_brightness", lambda x: (x > FUNDUS_BORDER_BRIGHTNESS_THRESHOLD).mean() * 100),
    )
    print(border_stats.round(2).to_string())
    print("=" * 72 + "\n")


def generate_eda_plots(df: pd.DataFrame, output_path: Path = EDA_SUMMARY_PNG) -> None:
    """Create multi-panel EDA figure with distribution plots and outlier thumbnails."""
    valid_df = df[~df["is_corrupt"]].copy()
    sns.set_theme(style="whitegrid", context="notebook")

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05])

    palette = {
        "cataract": "#E76F51",
        "diabetic_retinopathy": "#2A9D8F",
        "glaucoma": "#264653",
        "normal": "#E9C46A",
    }

    # --- Panel 1: Class Distribution per Split ---
    ax1 = fig.add_subplot(gs[0, 0])
    dist_counts = (
        valid_df.groupby(["split", "class_name"], observed=True)
        .size()
        .reset_index(name="count")
    )
    sns.barplot(
        data=dist_counts,
        x="split",
        y="count",
        hue="class_name",
        order=[s for s in SPLITS if s in valid_df["split"].unique()],
        hue_order=CLASSES,
        palette=palette,
        ax=ax1,
    )
    for container in ax1.containers:
        ax1.bar_label(container, fontsize=8, padding=2)
    ax1.set_title("Class Distribution by Split", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Split")
    ax1.set_ylabel("Image Count")
    ax1.legend(title="Class", fontsize=8, title_fontsize=9)

    # --- Panel 2: Border Brightness (Cataract vs Fundus Proof) ---
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=valid_df,
        x="class_name",
        y="border_brightness",
        order=CLASSES,
        hue="class_name",
        palette=palette,
        legend=False,
        fliersize=2.5,
        ax=ax2,
    )
    ax2.axhline(
        FUNDUS_BORDER_BRIGHTNESS_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"ROI Guard Threshold ({FUNDUS_BORDER_BRIGHTNESS_THRESHOLD})",
    )
    ax2.set_title("Border Brightness: Cataract vs. Fundus", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Mean Outer 5% Border Luminance (0-255)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(fontsize=8)

    # --- Panel 3: Blur Score (Laplacian Variance) ---
    ax3 = fig.add_subplot(gs[0, 2])
    valid_df["log_lap_var"] = np.log10(valid_df["laplacian_var"].clip(lower=0.1))
    sns.violinplot(
        data=valid_df,
        x="class_name",
        y="log_lap_var",
        order=CLASSES,
        hue="class_name",
        palette=palette,
        legend=False,
        inner="quartile",
        ax=ax3,
    )
    ax3.axhline(
        np.log10(BLUR_LAPLACIAN_THRESHOLD),
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Blur Threshold (log10({BLUR_LAPLACIAN_THRESHOLD}) = {np.log10(BLUR_LAPLACIAN_THRESHOLD):.2f})",
    )
    ax3.set_title("Blur Metric Distribution (log10 Laplacian Var)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("log10(Laplacian Variance)")
    ax3.tick_params(axis="x", rotation=15)
    ax3.legend(fontsize=8)

    # --- Panel 4: Image Dimensions Scatter ---
    ax4 = fig.add_subplot(gs[1, 0])
    sample_scatter = valid_df.sample(n=min(2000, len(valid_df)), random_state=42)
    sns.scatterplot(
        data=sample_scatter,
        x="width",
        y="height",
        hue="class_name",
        hue_order=CLASSES,
        palette=palette,
        alpha=0.55,
        s=30,
        ax=ax4,
    )
    ax4.set_title("Image Resolution Distribution (Sample)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Width (px)")
    ax4.set_ylabel("Height (px)")
    ax4.legend(title="Class", fontsize=8, title_fontsize=9)

    # --- Panel 5: Augmented vs Original Proportions ---
    ax5 = fig.add_subplot(gs[1, 1])
    aug_counts = (
        valid_df.groupby(["class_name", "is_augmented"], observed=True)
        .size()
        .reset_index(name="count")
    )
    aug_counts["type"] = aug_counts["is_augmented"].map({True: "Pre-Augmented (aug_)", False: "Original"})
    sns.barplot(
        data=aug_counts,
        x="class_name",
        y="count",
        hue="type",
        order=CLASSES,
        palette={"Original": "#457B9D", "Pre-Augmented (aug_)": "#F4A261"},
        ax=ax5,
    )
    for container in ax5.containers:
        ax5.bar_label(container, fontsize=8, padding=2)
    ax5.set_title("Original vs. Pre-Augmented Filenames", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Class")
    ax5.set_ylabel("Count")
    ax5.tick_params(axis="x", rotation=15)

    # --- Panel 6: Border vs Center Brightness Scatter ---
    ax6 = fig.add_subplot(gs[1, 2])
    sns.scatterplot(
        data=sample_scatter,
        x="border_brightness",
        y="center_brightness",
        hue="class_name",
        hue_order=CLASSES,
        palette=palette,
        alpha=0.55,
        s=25,
        ax=ax6,
    )
    ax6.axvline(FUNDUS_BORDER_BRIGHTNESS_THRESHOLD, color="red", linestyle="--", linewidth=1.2)
    ax6.set_title("Border vs. Center Brightness", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Outer Border Brightness")
    ax6.set_ylabel("Center Brightness")
    ax6.legend(title="Class", fontsize=8, title_fontsize=9)

    # --- Row 3: Outlier Sample Gallery (6 Sub-panels spanning bottom row) ---
    outlier_specs = []
    # 1. Most blurry overall
    outlier_specs.append(("Most Blurry Sample", valid_df.sort_values("laplacian_var").iloc[0]))
    # 2. Sharpest overall
    outlier_specs.append(("Sharpest Sample", valid_df.sort_values("laplacian_var", ascending=False).iloc[0]))
    # 3. Typical Cataract (median border)
    cat_df = valid_df[valid_df["class_name"] == "cataract"]
    if not cat_df.empty:
        cat_med_idx = (cat_df["border_brightness"] - cat_df["border_brightness"].median()).abs().idxmin()
        outlier_specs.append(("Cataract (Median Border)", valid_df.loc[cat_med_idx]))
        outlier_specs.append(("Cataract Outlier (Darkest Border)", cat_df.sort_values("border_brightness").iloc[0]))
    # 5. Typical Fundus (median border) & Outlier Fundus (brightest border)
    fundus_df = valid_df[valid_df["class_name"] != "cataract"]
    if not fundus_df.empty:
        fun_med_idx = (fundus_df["border_brightness"] - fundus_df["border_brightness"].median()).abs().idxmin()
        outlier_specs.append(("Fundus (Median Border)", valid_df.loc[fun_med_idx]))
        outlier_specs.append(("Fundus Outlier (Brightest Border)", fundus_df.sort_values("border_brightness", ascending=False).iloc[0]))

    gs_bottom = gs[2, :].subgridspec(1, len(outlier_specs))
    for i, (title, row) in enumerate(outlier_specs):
        ax_img = fig.add_subplot(gs_bottom[0, i])
        img = cv2.imread(row["filepath"])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax_img.imshow(img_rgb)
        ax_img.set_title(
            f"{title}\n{row['split']}/{row['class_name']}\n"
            f"Border={row['border_brightness']:.1f} | LapVar={row['laplacian_var']:.1f}",
            fontsize=8.5,
            fontweight="bold",
        )
        ax_img.axis("off")

    fig.suptitle("Iris Vision - Dataset Audit & Exploratory Data Analysis", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] EDA visualization saved to: {output_path}")


def main() -> None:
    ensure_output_dir()
    audit_df = run_dataset_audit(DATASET_DIR)
    audit_df.to_csv(AUDIT_REPORT_CSV, index=False)
    print(f"[SAVED] Audit report CSV saved to: {AUDIT_REPORT_CSV}")

    print_audit_summary(audit_df)
    generate_eda_plots(audit_df, EDA_SUMMARY_PNG)


if __name__ == "__main__":
    main()

