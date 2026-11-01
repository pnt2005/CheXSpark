"""Trích feature vectors 1024-d từ DenseNet121 (Stage-2 checkpoint) cho Gold layer.

Đầu vào : data/kaggle/stage2-nih-dev/stage2_train.csv  (labels + RLE mask + split)
          data/bronze/images/<image_index>              (2k ảnh PNG)
          notebooks/kaggle/best_stage2_densenet121.pth  (state_dict, classifier 14 lớp)

Đầu ra  : data/tmp/densenet_features.parquet
          (image_index, patient_id, split, label_* x14, feat_0000..feat_1023)

Preprocessing **giống hệt lúc train**: grayscale → resize 224 → nhân lung mask
(MASK_MODE='lungs') → 3 kênh → ImageNet normalize.

Sau khi chạy, upload Parquet lên MinIO `gold/features/densenet121/` để Spark MLlib
(PCA / K-Means / LR / RF) đọc.

Usage:
    python scripts/dev/extract_densenet_features.py
    python scripts/dev/extract_densenet_features.py --mask-mode none --out data/tmp/features_none.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_SIZE = 224
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def decode_rle(rle, h: int, w: int) -> np.ndarray:
    """CheXmask RLE: chuỗi cặp (start, length), reshape row-major (C order)."""
    vals = [int(v) for v in str(rle).split()]
    flat = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(vals), 2):
        flat[vals[i]: vals[i] + vals[i + 1]] = 1
    return flat.reshape((h, w))


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, images_dir: Path, mask_mode: str):
        self.df = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return len(self.df)

    def _mask(self, row) -> np.ndarray:
        h, w = int(row["mask_height"]), int(row["mask_width"])
        m = decode_rle(row["mask_left_rle"], h, w) | decode_rle(row["mask_right_rle"], h, w)
        if self.mask_mode == "lungs_heart":
            m = m | decode_rle(row["mask_heart_rle"], h, w)
        m = Image.fromarray((m * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.NEAREST
        )
        return (np.array(m) > 127).astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / row["image_index"]).convert("L").resize(
            (IMG_SIZE, IMG_SIZE)
        )
        arr = np.array(img).astype(np.float32) / 255.0
        if self.mask_mode != "none":
            arr = arr * self._mask(row)
        x = np.stack([arr] * 3, axis=0)
        return NORM(torch.from_numpy(x)), idx


def build_feature_extractor(ckpt: Path) -> nn.Module:
    """DenseNet121 với classifier 14 lớp; bỏ classifier → output 1024-d global pool."""
    model = models.densenet121(weights=None)  # state_dict đã có đủ trọng số
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def extract(model, loader, n: int):
    feats = np.zeros((n, 1024), dtype=np.float32)
    done = 0
    t0 = time.time()
    for x, idxs in loader:
        f = model.features(x)
        f = F.relu(f)
        f = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)  # (B, 1024)
        feats[idxs.numpy()] = f.numpy()
        done += len(idxs)
        if done % 320 == 0 or done == n:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{n} ảnh ({rate:.0f} img/s)")
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("data/kaggle/stage2-nih-dev/stage2_train.csv"))
    ap.add_argument("--images-dir", type=Path, default=Path("data/bronze/images"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("notebooks/kaggle/best_stage2_densenet121.pth"))
    ap.add_argument("--out", type=Path, default=Path("data/tmp/densenet_features.parquet"))
    ap.add_argument("--mask-mode", default="lungs", choices=["lungs", "lungs_heart", "none"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Rows: {len(df)} | mask_mode={args.mask_mode} | ckpt={args.ckpt.name}")

    model = build_feature_extractor(args.ckpt)
    ds = FeatureDataset(df, args.images_dir, args.mask_mode)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)

    feats = extract(model, dl, len(df))
    print(f"Feature matrix: {feats.shape} | mean={feats.mean():.4f} std={feats.std():.4f}")

    feat_cols = [f"feat_{i:04d}" for i in range(feats.shape[1])]
    out = pd.concat(
        [
            df[["image_index", "patient_id", "split"] + [f"label_{d}" for d in LABELS]],
            pd.DataFrame(feats, columns=feat_cols),
        ],
        axis=1,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Đã ghi {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MB)")
    print("Bước tiếp: upload lên MinIO gold/features/densenet121/ rồi chạy src/gold/mllib_analysis.py")


if __name__ == "__main__":
    main()
