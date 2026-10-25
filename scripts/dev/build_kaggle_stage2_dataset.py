"""Gộp Silver labels + CheXmask RLE thành 1 CSV train cho Kaggle Stage-2.

Input:
  - silver_labels.csv       : 2000 rows (nhãn 14 bệnh + split, từ Silver Parquet)
  - chexmask_silver2k.csv   : 2000 rows (RLE mask phổi/tim, đã filter từ CheXmask)

Output (thư mục Kaggle dataset):
  - stage2_train.csv        : join 2 bảng trên image_index
  - dataset-metadata.json   : metadata Kaggle API
  - dataset_description.md  : mô tả dataset

Masks là RLE 1024×1024 theo cặp (start, length) — notebook sẽ decode rồi resize
về 224×224 khớp ảnh khanfashee.

Usage:
    python scripts/dev/build_kaggle_stage2_dataset.py \
        --silver-labels data/tmp/silver_labels.csv \
        --chexmask data/tmp/chexmask_silver2k.csv \
        --out-dir data/kaggle/stage2-nih-dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLS_PREFIX = "label_"
KEEP_SILVER = [
    "image_index",
    "patient_id",
    "finding_labels",
    "patient_age",
    "patient_gender",
    "view_position",
    "n_diseases",
    "no_finding",
    "split",
]
KEEP_MASK = [
    "Image Index",
    "Dice RCA (Mean)",
    "Dice RCA (Max)",
    "Left Lung",
    "Right Lung",
    "Heart",
    "Height",
    "Width",
]


def decode_rle(rle_str: str, height: int, width: int) -> np.ndarray:
    """Decode RLE dạng cặp (start, length), reshape row-major (C order).

    Đã xác nhận qua validate bounding-box vs Landmarks: order='C' đúng.
    """
    vals = [int(v) for v in str(rle_str).split()]
    flat = np.zeros(height * width, dtype=np.uint8)
    for i in range(0, len(vals), 2):
        start, length = vals[i], vals[i + 1]
        flat[start : start + length] = 1
    return flat.reshape((height, width))


def validate_masks(df: pd.DataFrame, n_samples: int = 5) -> None:
    """Decode n mẫu mask, kiểm tra kích thước + số pixel phổi hợp lý."""
    print(f"Validate {n_samples} mask ngẫu nhiên:")
    sample = df.sample(n=min(n_samples, len(df)), random_state=0)
    for _, row in sample.iterrows():
        h, w = int(row["Height"]), int(row["Width"])
        total_fg = 0
        for col in ["Left Lung", "Right Lung", "Heart"]:
            mask = decode_rle(row[col], h, w)
            assert mask.shape == (h, w), f"mask shape {mask.shape} != {(h, w)}"
            total_fg += int(mask.sum())
        frac = total_fg / (h * w)
        status = "OK" if 0.05 < frac < 0.75 else "CẢNH BÁO"
        print(
            f"  {row['image_index']}: {h}x{w}, fg={total_fg:,} ({frac:.1%}) [{status}]"
        )


def build(silver_path: Path, chexmask_path: Path, out_dir: Path) -> Path:
    silver = pd.read_csv(silver_path, dtype={"image_index": str})
    masks = pd.read_csv(chexmask_path, dtype={"Image Index": str})
    masks = masks.rename(columns={"Image Index": "image_index"})

    label_cols = [c for c in silver.columns if c.startswith(LABEL_COLS_PREFIX)]
    mask_cols = ["image_index"] + [
        c for c in KEEP_MASK if c in masks.columns and c != "Image Index"
    ]
    merged = silver[[c for c in KEEP_SILVER if c in silver.columns] + label_cols].merge(
        masks[mask_cols],
        on="image_index",
        how="left",
        validate="one_to_one",
    )

    # Kiểm tra không có mask thiếu (NaN)
    missing_mask = merged["Left Lung"].isna().sum()
    print(f"Rows: {len(merged)} | missing mask: {missing_mask}")
    if missing_mask:
        print("  CẢNH BÁO: có ảnh không có mask — cần xử lý trước khi train")

    validate_masks(merged.dropna(subset=["Left Lung"]).head(50))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stage2_train.csv"
    merged.to_csv(csv_path, index=False)
    print(f"Đã ghi {csv_path} ({csv_path.stat().st_size / (1024 * 1024):.1f} MB)")

    # Metadata Kaggle (slug lowercase, không khoảng trắng)
    metadata = {
        "title": "CheXSpark Stage2 NIH Dev (labels + CheXmask RLE)",
        "id": "chexspark-stage2-nih-dev",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    description = (
        "# CheXSpark Stage-2 NIH Dev Dataset\n\n"
        "Nhãn 14 bệnh + split (từ Silver layer) và mask phổi/tim RLE (CheXmask) "
        "cho 2,000 ảnh dev NIH ChestX-ray14.\n\n"
        "- `stage2_train.csv` — join labels + RLE masks theo `image_index`\n"
        "- 14 cột `label_*` (binary), `split` (train/val/test, patient-level)\n"
        "- Mask RLE 1024×1024 dạng cặp `(start, length)`, cột `Left Lung`, "
        "`Right Lung`, `Heart`, `Height`, `Width`\n\n"
        "Ảnh X-quang KHÔNG kèm theo — dùng chung dataset Kaggle "
        "`khanfashee/nih-chest-x-ray-14-224x224-resized` (join theo `image_index`).\n"
    )
    (out_dir / "dataset_description.md").write_text(description, encoding="utf-8")

    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-labels", type=Path, default=Path("data/tmp/silver_labels.csv"))
    ap.add_argument("--chexmask", type=Path, default=Path("data/tmp/chexmask_silver2k.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/kaggle/stage2-nih-dev"))
    args = ap.parse_args()
    build(args.silver_labels, args.chexmask, args.out_dir)


if __name__ == "__main__":
    main()
