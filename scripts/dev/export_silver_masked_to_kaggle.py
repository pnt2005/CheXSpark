"""Export Silver masked Parquet → stage2_train.csv (Kaggle) cho Stage-2.

Thay cho 2 script pandas cũ (filter CheXmask + build Kaggle dataset): giờ
toàn bộ labels + RLE masks đã nằm trong Silver `cxr14_masked` (join bằng
Spark job ingest_chexmask_to_silver.py) — script này chỉ copy Parquet về
local (qua mc cp), gộp partition và xuất CSV + Kaggle metadata.

Usage:
    # 1) Copy Parquet từ MinIO về local
    #    docker run ... mc cp --recursive local/silver/cxr14_masked /out/
    # 2) Export
    python scripts/dev/export_silver_masked_to_kaggle.py \
        --silver-dir data/tmp/silver_masked \
        --out-dir data/kaggle/stage2-nih-dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def decode_rle(rle_str: str, height: int, width: int) -> np.ndarray:
    """Decode RLE CheXmask dạng cặp (start, length), row-major (C order).

    Đã validate qua bounding-box vs Landmarks: order='C' đúng.
    """
    vals = [int(v) for v in str(rle_str).split()]
    flat = np.zeros(height * width, dtype=np.uint8)
    for i in range(0, len(vals), 2):
        start, length = vals[i], vals[i + 1]
        flat[start : start + length] = 1
    return flat.reshape((height, width))


def load_silver_masked(silver_dir: Path) -> pd.DataFrame:
    """Đọc Parquet partition theo split (cột split nằm trong đường dẫn).

    Dùng glob '**/split=*' cho linh hoạt — mc cp có thể để lại cấp thư mục
    tên bucket (ví dụ .../cxr14_masked/split=train/).
    """
    frames = []
    for split_dir in sorted(silver_dir.glob("**/split=*")):
        split_value = split_dir.name.split("=", 1)[1]
        for part_file in sorted(split_dir.glob("*.parquet")):
            df = pq.read_table(part_file).to_pandas()
            df["split"] = split_value
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Không tìm thấy Parquet trong {silver_dir}")
    return pd.concat(frames, ignore_index=True)


def validate_masks(df: pd.DataFrame, n_samples: int = 5) -> None:
    """Decode n mẫu mask — kiểm tra shape + foreground hợp lý (không cần ảnh)."""
    print(f"Validate {n_samples} mask ngẫu nhiên:")
    sample = df.sample(n=min(n_samples, len(df)), random_state=0)
    for _, row in sample.iterrows():
        h, w = int(row["mask_height"]), int(row["mask_width"])
        total_fg = 0
        for col in ["mask_left_rle", "mask_right_rle", "mask_heart_rle"]:
            mask = decode_rle(row[col], h, w)
            assert mask.shape == (h, w), f"mask shape {mask.shape} != {(h, w)}"
            total_fg += int(mask.sum())
        frac = total_fg / (h * w)
        status = "OK" if 0.05 < frac < 0.75 else "CẢNH BÁO"
        print(f"  {row['image_index']}: {h}x{w}, fg={total_fg:,} ({frac:.1%}) [{status}]")


def build(silver_dir: Path, out_dir: Path) -> Path:
    df = load_silver_masked(silver_dir)

    # Kiểm tra mask đầy đủ (không NaN/null)
    missing = int(df["mask_left_rle"].isna().sum() + df["mask_right_rle"].isna().sum())
    print(f"Rows: {len(df)} | missing mask cells: {missing}")
    if missing:
        print("  CẢNH BÁO: có mask thiếu — xử lý trước khi train")

    print("Per split:")
    print(df["split"].value_counts().to_string())
    validate_masks(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stage2_train.csv"
    df.to_csv(csv_path, index=False)
    print(f"Đã ghi {csv_path} ({csv_path.stat().st_size / (1024 * 1024):.1f} MB)")

    metadata = {
        "title": "CheXSpark Stage2 NIH Dev (Silver masked: labels + CheXmask RLE)",
        "id": "chexspark-stage2-nih-dev",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    description = (
        "# CheXSpark Stage-2 NIH Dev Dataset\n\n"
        "Xuất trực tiếp từ Silver layer `cxr14_masked` (join CheXmask bằng Spark) — "
        "2,000 ảnh NIH ChestX-ray14 dev subset.\n\n"
        "- `stage2_train.csv` — labels 14 bệnh + split + RLE masks trong 1 file\n"
        "- 14 cột `label_*` (binary), `split` (patient-level train/val/test)\n"
        "- Mask: `mask_left_rle`, `mask_right_rle`, `mask_heart_rle` "
        "(RLE cặp (start,length), 1024×1024, cột `mask_height`/`mask_width`)\n"
        "- `dice_rca_mean`/`dice_rca_max` — chất lượng mask (ngưỡng khuyến nghị ≥ 0.7)\n\n"
        "Ảnh X-quang KHÔNG kèm — dùng Kaggle dataset "
        "`khanfashee/nih-chest-x-ray-14-224x224-resized` (join theo `image_index`).\n"
    )
    (out_dir / "dataset_description.md").write_text(description, encoding="utf-8")
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir", type=Path, default=Path("data/tmp/silver_masked"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/kaggle/stage2-nih-dev"))
    args = ap.parse_args()
    build(args.silver_dir, args.out_dir)


if __name__ == "__main__":
    main()
