"""Filter CheXmask (ChestX-Ray8.csv) xuống đúng tập ảnh dev Silver (2k ảnh).

CheXmask OriginalResolution/ChestX-Ray8.csv (~2.04GB) chứa RLE mask cho ~112k ảnh
NIH nhưng KHÔNG sort theo Image Index tăng dần, nên phải đọc toàn bộ rồi lọc.

Chỉ giữ các cột cần cho segmentation (bỏ 'Landmarks' — chuỗi tọa độ cực lớn,
không dùng để tạo mask) để thu file Kaggle xuống vài MB.

Usage:
    python scripts/dev/filter_chexmask_to_silver.py \
        --chexmask data/raw/chexmask/ChestX-Ray8.csv \
        --silver-labels data/tmp/silver_labels.csv \
        --out data/tmp/chexmask_silver2k.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Cột CheXmask cần giữ (bỏ Landmarks cho nhẹ)
KEEP_COLS = [
    "Image Index",
    "Dice RCA (Mean)",
    "Dice RCA (Max)",
    "Left Lung",
    "Right Lung",
    "Heart",
    "Height",
    "Width",
]
CHUNK_ROWS = 100_000


def filter_chexmask(
    chexmask_path: Path, silver_labels_path: Path, out_path: Path
) -> int:
    silver = pd.read_csv(silver_labels_path, dtype={"image_index": str})
    target_ids = set(silver["image_index"])
    print(f"Silver dev images cần mask: {len(target_ids)}")

    matched_frames: list[pd.DataFrame] = []
    scanned = 0
    reader = pd.read_csv(
        chexmask_path,
        usecols=KEEP_COLS,
        chunksize=CHUNK_ROWS,
        dtype={"Image Index": str},
        low_memory=False,
    )
    for chunk in reader:
        scanned += len(chunk)
        hit = chunk[chunk["Image Index"].isin(target_ids)]
        if not hit.empty:
            matched_frames.append(hit)
        if scanned % 1_000_000 == 0:
            print(f"  ...đã quét {scanned:,} rows, khớp {sum(len(m) for m in matched_frames):,}")

    if not matched_frames:
        raise RuntimeError("Không khớp dòng nào — kiểm tra Image Index format!")

    result = pd.concat(matched_frames, ignore_index=True)
    missing = target_ids - set(result["Image Index"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"Đã quét {scanned:,} rows CheXmask")
    print(f"Khớp {len(result):,}/{len(target_ids)} ảnh dev → {out_path}")
    print(f"Thiếu mask: {len(missing)} ảnh")
    if missing:
        sample = sorted(missing)[:10]
        print(f"  ví dụ thiếu: {sample}")
    print(f"Kích thước file: {out_path.stat().st_size / (1024 * 1024):.1f} MB")
    return len(result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chexmask", type=Path, default=Path("data/raw/chexmask/ChestX-Ray8.csv"))
    ap.add_argument("--silver-labels", type=Path, default=Path("data/tmp/silver_labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/tmp/chexmask_silver2k.csv"))
    args = ap.parse_args()
    filter_chexmask(args.chexmask, args.silver_labels, args.out)


if __name__ == "__main__":
    main()
