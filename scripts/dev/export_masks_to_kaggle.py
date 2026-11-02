"""Silver precomputed masks → Kaggle dataset `chexspark-masks-224` (packed .npy).

Đầu vào là output của `src/silver/precompute_masks.py` (Parquet partition theo
`split`, mỗi dòng có `mask_lung_bits` 6272 byte + `png_lung`). Script này gom
thành **một file .npy/ split** để Kaggle đọc bằng `mmap` (không decode, không
tốn RAM):

    masks_train.npy   uint8 (n, 6272)  = np.packbits(mask 224×224)
    masks_val.npy
    masks_test.npy
    mask_index.csv    split,image_index,mask_row,n_fg_lung,n_fg_heart,dice_rca_mean

Vì sao .npy packed thay vì 112k file PNG: Kaggle giới hạn ~100k file/dataset và
mount 112k file nhỏ cực chậm; 3 file .npy (~0,7 GB cho 112k ảnh) đọc bằng memmap
gần như tức thời (`np.unpackbits` ~0,1 ms/ảnh, nhanh hơn decode RLE ~100×).

Usage:
    # 1) Copy Parquet từ MinIO về local (partition split=* nằm trong đường dẫn)
    docker run --rm --network chexspark_chexspark-net -v "%CD%\\data\\tmp:/out" \\
        minio/mc cp --recursive local/silver/cxr14_masks /out/silver_masks

    # 2) Export
    python scripts/dev/export_masks_to_kaggle.py \\
        --silver-dir data/tmp/silver_masks --out-dir data/kaggle/stage2-masks-224

    # (tuỳ chọn) xả PNG ra đĩa để xem bằng mắt
    python scripts/dev/export_masks_to_kaggle.py --silver-dir ... --write-pngs data/tmp/mask_pngs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

IMG_SIZE = 224
PACK_BYTES = IMG_SIZE * IMG_SIZE // 8  # 6272
SPLITS = ("train", "val", "test")


def load_masks(silver_dir: Path) -> dict[str, pd.DataFrame]:
    """Đọc Parquet của precompute job, giữ nguyên cột split theo tên thư mục."""
    per_split: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}
    for split_dir in sorted(silver_dir.glob("**/split=*")):
        split = split_dir.name.split("=", 1)[1]
        if split not in per_split:
            continue
        for f in sorted(split_dir.glob("*.parquet")):
            per_split[split].append(pd.read_parquet(f))
    result = {}
    for split, frames in per_split.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["split"] = split
        result[split] = df.sort_values("image_index").reset_index(drop=True)
    if not result:
        raise FileNotFoundError(f"Không thấy Parquet split=* trong {silver_dir}")
    return result


def unpack(bits: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bits, dtype=np.uint8)).reshape(IMG_SIZE, IMG_SIZE)


def validate(df: pd.DataFrame, n: int = 5) -> None:
    """Giải nén vài mask: shape đúng, số pixel foreground khớp cột n_fg_lung."""
    print(f"  validate {n} mask:")
    for _, row in df.sample(n=min(n, len(df)), random_state=0).iterrows():
        m = unpack(row["mask_lung_bits"])
        assert m.shape == (IMG_SIZE, IMG_SIZE), m.shape
        assert m.dtype == np.uint8
        got, want = int(m.sum()), int(row["n_fg_lung"])
        assert got == want, f"{row['image_index']}: fg {got} != {want}"
        frac = got / (IMG_SIZE * IMG_SIZE)
        flag = "OK" if 0.03 < frac < 0.60 else "CẢNH BÁO"
        print(f"    {row['image_index']}: fg={got:,} ({frac:.1%}) [{flag}]")


def build(silver_dir: Path, out_dir: Path, write_pngs: Path | None = None) -> None:
    data = load_masks(silver_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    for split in SPLITS:
        df = data.get(split)
        if df is None:
            print(f"[WARN] thiếu split '{split}' — bỏ qua")
            continue

        print(f"\n[{split}] {len(df)} mask")
        validate(df)

        for kind, col in (("", "mask_lung_bits"), ("heart_", "mask_heart_bits")):
            pack = np.stack([
                np.frombuffer(b, dtype=np.uint8).copy() for b in df[col]
            ]).astype(np.uint8)
            assert pack.shape == (len(df), PACK_BYTES), pack.shape

            npy_path = out_dir / f"masks_{kind}{split}.npy"
            np.save(npy_path, pack)
            print(f"  -> {npy_path.name} {pack.shape} "
                  f"{npy_path.stat().st_size / 1024**2:.1f} MB")

        for row_idx, row in enumerate(df.itertuples(index=False)):
            index_rows.append({
                "split": split,
                "image_index": row.image_index,
                "mask_row": row_idx,
                "n_fg_lung": int(row.n_fg_lung),
                "n_fg_heart": int(row.n_fg_heart),
                "dice_rca_mean": row.dice_rca_mean,
            })

        if write_pngs is not None:
            png_dir = write_pngs / split
            png_dir.mkdir(parents=True, exist_ok=True)
            for row, bits in zip(df.itertuples(index=False), df["png_lung"]):
                (png_dir / f"{row.image_index}.png").write_bytes(bits)
            print(f"  -> {len(df)} PNG vào {png_dir}")

    index = pd.DataFrame(index_rows)
    index_path = out_dir / "mask_index.csv"
    index.to_csv(index_path, index=False)
    print(f"\nmask_index.csv: {len(index)} dòng")

    total_mb = sum(p.stat().st_size for p in out_dir.glob("*.npy")) / 1024**2
    print(f"Tổng masks .npy: {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)")

    metadata = {
        "title": "CheXSpark Masks 224 (packed)",
        "id": "chexspark-masks-224",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (out_dir / "dataset_description.md").write_text(
        "# CheXSpark Masks 224 — mask phổi CheXmask đã precompute\n\n"
        "Sinh bằng Spark (`src/silver/precompute_masks.py`) từ `silver/cxr14_masked`:\n"
        "decode RLE 1024×1024 → OR left|right → resize 224×224 (NEAREST) → packbits.\n\n"
        "- `masks_{split}.npy` — `uint8 (n, 6272)` = `np.packbits(mask phổi 224×224)`\n"
        "  (`.reshape(224,224)` sau `np.unpackbits`)\n"
        "- `masks_heart_{split}.npy` — cùng định dạng, mask **tim** (OR vào khi\n"
        "  `MASK_MODE='lungs_heart'`)\n"
        "- `mask_index.csv` — `split,image_index,mask_row,...`; dòng `mask_row` của\n"
        "  `masks_{split}.npy` tương ứng `image_index`\n"
        "- `n_fg_lung` / `n_fg_heart` — số pixel (kiểm tra nhanh, không cần giải nén)\n\n"
        "Dùng trong notebook Kaggle: `np.load(..., mmap_mode='r')` → ~0,1 ms/ảnh,\n"
        "thay cho việc decode RLE mỗi epoch (~8–12 ms/ảnh).\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir", type=Path, default=Path("data/tmp/silver_masks"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/kaggle/stage2-masks-224"))
    ap.add_argument("--write-pngs", type=Path, default=None,
                    help="xả PNG 224×224 ra thư mục này (xem bằng mắt)")
    args = ap.parse_args()
    build(args.silver_dir, args.out_dir, args.write_pngs)


if __name__ == "__main__":
    main()
