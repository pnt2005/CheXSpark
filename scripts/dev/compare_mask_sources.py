"""Ablation: mask phổi CheXmask (GT) vs U-Net Stage-1 tự sinh → Stage-2 có lệch không?

Đây là câu hỏi triển khai quan trọng: nếu U-Net sinh mask khác CheXmask một chút thì
xác suất 14 bệnh và AUC có tụt không? Script chạy DenseNet121 **hai lần** trên cùng
tập test, chỉ đổi nguồn mask, rồi báo:

  - mean |Δp| (trung bình toàn bộ 344×14 xác suất) và theo từng bệnh
  - AUC từng bệnh với mỗi nguồn mask → chênh lệch mAUROC

Usage:
    python scripts/dev/compare_mask_sources.py --split test
    python scripts/dev/compare_mask_sources.py --from-registry --stage1-from-registry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dl.model import load_stage2_model  # noqa: E402
from dl.preprocess import DISEASES, LABEL_COLS, build_tensor, load_gray, lung_mask  # noqa: E402
from dl.segmentation import load_stage1_model, predict_mask  # noqa: E402


@torch.no_grad()
def predict(model, df: pd.DataFrame, images_dir: Path, device: str, mask_fn, batch_size: int = 32):
    probs = np.zeros((len(df), len(DISEASES)), dtype=np.float32)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        tensors = []
        for _, row in chunk.iterrows():
            gray = load_gray(images_dir / row["image_index"])
            tensors.append(build_tensor(gray, mask_fn(row, gray)))
        x = torch.cat(tensors).to(device)
        probs[start:start + len(chunk)] = torch.sigmoid(model(x)).cpu().numpy()
        if (start // batch_size) % 5 == 0:
            print(f"  {min(start + batch_size, len(df))}/{len(df)}")
    return probs


def per_label_auc(y: np.ndarray, p: np.ndarray) -> list[float]:
    out = []
    for i in range(y.shape[1]):
        out.append(roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "data/kaggle/stage2-nih-dev/stage2_train.csv")
    ap.add_argument("--images-dir", type=Path, default=ROOT / "data/bronze/images")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "notebooks/kaggle/best_stage2_densenet121.pth")
    ap.add_argument("--stage1-ckpt", type=Path,
                    default=ROOT / "notebooks/kaggle/best_stage1_unet.pth")
    ap.add_argument("--from-registry", action="store_true")
    ap.add_argument("--stage1-from-registry", action="store_true")
    ap.add_argument("--alias", default="production")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/outputs/stage1")
    args = ap.parse_args()

    if args.from_registry:
        from mlops.registry import load_stage2_from_registry

        model = load_stage2_from_registry(alias=args.alias, device=args.device)
    else:
        model = load_stage2_model(args.ckpt, args.device)

    if args.stage1_from_registry:
        from mlops.registry import load_stage1_from_registry

        seg = load_stage1_from_registry(alias=args.alias, device=args.device)
    else:
        seg = load_stage1_model(args.stage1_ckpt, args.device)

    df = pd.read_csv(args.csv, dtype={"image_index": str})
    df = df[(df.split == args.split) & df["mask_left_rle"].notna()].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    y = df[LABEL_COLS].to_numpy().astype(np.int8)
    print(f"{args.split}: {len(df)} ảnh — chạy 2 lượt (CheXmask vs U-Net)")

    print("[1/2] mask CheXmask (ground truth)")
    p_gt = predict(model, df, args.images_dir, args.device, lambda row, gray: lung_mask(row))
    print("[2/2] mask U-Net Stage-1")
    p_unet = predict(model, df, args.images_dir, args.device,
                     lambda row, gray: predict_mask(seg, gray, args.device))

    auc_gt = per_label_auc(y, p_gt)
    auc_unet = per_label_auc(y, p_unet)
    delta = np.abs(p_gt - p_unet)

    print("\n=== KẾT QUẢ ===")
    print(f"mean |Δp| toàn bộ      : {delta.mean():.4f}")
    print(f"mean |Δp| lớn nhất bệnh: {delta.mean(0).max():.4f}")
    print(f"mAUROC CheXmask        : {np.nanmean(auc_gt):.4f}")
    print(f"mAUROC U-Net           : {np.nanmean(auc_unet):.4f}")
    print(f"chênh lệch mAUROC      : {np.nanmean(auc_unet) - np.nanmean(auc_gt):+.4f}")

    res = pd.DataFrame({
        "disease": DISEASES,
        "auc_chexmask": auc_gt,
        "auc_unet": auc_unet,
        "delta_auc": np.array(auc_unet) - np.array(auc_gt),
        "mean_abs_delta_p": delta.mean(0),
    }).sort_values("mean_abs_delta_p", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"mask_source_ablation_{args.split}.csv"
    res.to_csv(out, index=False)
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
