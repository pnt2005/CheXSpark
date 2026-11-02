"""Export Silver Parquet (partition theo split) → 1 CSV phẳng cho Kaggle.

Spark partitionBy('split') đẩy cột 'split' vào đường dẫn chứ không nằm trong file
Parquet, nên ta phải đọc từng thư mục split=* và thêm lại cột split.

Usage:
    python scripts/dev/export_silver_labels.py \
        --silver-dir data/tmp/silver/cxr14 \
        --out data/tmp/silver_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def export(silver_dir: Path, out_path: Path, drop: list[str] | None = None) -> int:
    frames = []
    for split_dir in sorted(silver_dir.glob("split=*")):
        split_value = split_dir.name.split("=", 1)[1]
        for part_file in sorted(split_dir.glob("*.parquet")):
            df = pq.read_table(part_file).to_pandas()
            df["split"] = split_value  # thêm lại cột split (bị partition tách ra)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Không tìm thấy Parquet trong {silver_dir}")

    merged = pd.concat(frames, ignore_index=True)
    if drop:
        cols = [c for c in drop if c in merged.columns]
        merged = merged.drop(columns=cols)
        print(f"Đã bỏ cột: {cols}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # Thống kê nhanh
    print(f"Exported {len(merged)} rows -> {out_path}")
    print("Per split:")
    print(merged["split"].value_counts().to_string())
    return len(merged)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir", type=Path, default=Path("data/tmp/silver/cxr14"))
    ap.add_argument("--out", type=Path, default=Path("data/tmp/silver_labels.csv"))
    ap.add_argument("--drop", nargs="*", default=[],
                    help="cột cần loại khỏi CSV (vd: image_uri silver_ingested_at)")
    args = ap.parse_args()
    export(args.silver_dir, args.out, args.drop)


if __name__ == "__main__":
    main()
