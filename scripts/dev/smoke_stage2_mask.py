"""Smoke test logic mask Stage-2 bằng data thật local (không cần torch).

Replicate chính xác decode_rle + mask x anh trong notebook, chạy tren 3 anh bronze
de xac nhan mask decode dung, kich thuoc va ty le phu hop ly.

Usage: python scripts/dev/smoke_stage2_mask.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

STAGE2_CSV = Path("data/kaggle/stage2-nih-dev/stage2_train.csv")
BRONZE_IMG = Path("data/bronze/images")
IMG_SIZE = 224


def decode_rle(rle: str, h: int, w: int) -> np.ndarray:
    vals = [int(v) for v in str(rle).split()]
    flat = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(vals), 2):
        flat[vals[i] : vals[i] + vals[i + 1]] = 1
    return flat.reshape((h, w))


def main() -> None:
    df = pd.read_csv(STAGE2_CSV, dtype={"image_index": str}).head(3)
    ok = 0
    for _, r in df.iterrows():
        img_path = BRONZE_IMG / r["image_index"]
        if not img_path.exists():
            print(f"  MISS anh: {r['image_index']}")
            continue
        img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img_a = np.array(img).astype(np.float32) / 255.0
        h, w = int(r["mask_height"]), int(r["mask_width"])
        lung = decode_rle(r["mask_left_rle"], h, w) | decode_rle(r["mask_right_rle"], h, w)
        m = (
            np.array(
                Image.fromarray((lung * 255).astype(np.uint8)).resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST
                )
            )
            > 127
        )
        masked = img_a * m.astype(np.float32)
        print(
            f"{r['image_index']}: img{img_a.shape} cover={m.mean():.1%} "
            f"masked_mean={masked.mean():.3f} fg_px={m.sum()}"
        )
        ok += 1
    print("SMOKE TEST OK" if ok > 0 else "NO IMAGES TESTED")


if __name__ == "__main__":
    main()
