"""Stage-1: train U-Net phân đoạn phổi trên Silver masked (CheXmask ground truth).

Dùng **cùng** dữ liệu với Stage-2: `stage2_train.csv` (labels + RLE mask 1024×1024)
+ ảnh 224×224. Mask GT = `mask_left_rle | mask_right_rle` (không gồm tim).

Chạy được cả CPU (dev subset) và GPU (Kaggle/Colab). Trên Kaggle dùng notebook
`notebooks/kaggle/stage1_unet_segmentation.ipynb`; script này để train/eval nhanh
ở máy local hoặc trong CI.

Usage:
    # Smoke test nhanh trên CPU (300 ảnh, 3 epoch)
    python scripts/dev/train_stage1_unet.py --limit 300 --epochs 3

    # Train đầy đủ trên GPU
    python scripts/dev/train_stage1_unet.py --epochs 30 --batch-size 16

Output:
    notebooks/kaggle/best_stage1_unet.pth    checkpoint tốt nhất theo val Dice
    notebooks/kaggle/stage1_metrics.json     lịch sử train + Dice/IoU val/test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dl.preprocess import IMG_SIZE, decode_rle  # noqa: E402
from dl.segmentation import BCEDiceLoss, build_unet, dice_coef, iou_score, to_tensor  # noqa: E402


class LungSegDataset(Dataset):
    """(ảnh xám 224², mask phổi 224²) — mask decode từ CheXmask RLE và cache RAM."""

    def __init__(self, frame: pd.DataFrame, images_dir: Path, img_size: int = IMG_SIZE,
                 train: bool = False) -> None:
        self.df = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.img_size = img_size
        self.train = train
        self._masks: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _mask(self, idx: int) -> np.ndarray:
        """GT mask 224² (0/1) — decode 1 lần rồi giữ trong RAM."""
        if idx in self._masks:
            return self._masks[idx]
        row = self.df.iloc[idx]
        h, w = int(row["mask_height"]), int(row["mask_width"])
        m = decode_rle(row["mask_left_rle"], h, w) | decode_rle(row["mask_right_rle"], h, w)
        m = Image.fromarray((m * 255).astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )
        arr = (np.array(m) > 127).astype(np.float32)
        self._masks[idx] = arr
        return arr

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / row["image_index"]).convert("L").resize(
            (self.img_size, self.img_size)
        )
        gray = np.array(img).astype(np.float32) / 255.0
        mask = self._mask(idx)

        if self.train and np.random.rand() < 0.5:  # lật ngang: phổi đối xứng, nhãn không đổi
            gray = gray[:, ::-1].copy()
            mask = mask[:, ::-1].copy()

        return to_tensor(gray)[0], torch.from_numpy(mask)[None]


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: str) -> dict:
    model.eval()
    dices, ious = [], []
    for x, y in dl:
        prob = torch.sigmoid(model(x.to(device))).cpu().numpy()[:, 0]
        gt = y.numpy()[:, 0]
        for p, g in zip(prob, gt):
            pb = (p > 0.5).astype(np.float32)
            dices.append(dice_coef(pb, g))
            ious.append(iou_score(pb, g))
    d = np.array(dices)
    i = np.array(ious)
    return {
        "dice_mean": float(d.mean()),
        "dice_median": float(np.median(d)),
        "dice_p10": float(np.percentile(d, 10)),
        "iou_mean": float(i.mean()),
        "frac_dice_ge_07": float((d >= 0.7).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "data/kaggle/stage2-nih-dev/stage2_train.csv")
    ap.add_argument("--images-dir", type=Path, default=ROOT / "data/bronze/images")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "notebooks/kaggle")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--base", type=int, default=16, help="số kênh tầng đầu của U-Net")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số ảnh train (0 = tất cả)")
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0, help="giới hạn số ảnh test khi eval cuối")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(12, torch.get_num_threads()))

    df = pd.read_csv(args.csv, dtype={"image_index": str})
    df = df[df["mask_left_rle"].notna()].reset_index(drop=True)
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=args.seed)
    if args.val_limit:
        val_df = val_df.sample(n=min(args.val_limit, len(val_df)), random_state=args.seed)
    if args.test_limit:
        test_df = test_df.sample(n=min(args.test_limit, len(test_df)), random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)} | "
          f"img={args.img_size} base={args.base} device={args.device}")

    tr_ds = LungSegDataset(train_df, args.images_dir, args.img_size, train=True)
    va_ds = LungSegDataset(val_df, args.images_dir, args.img_size)
    te_ds = LungSegDataset(test_df, args.images_dir, args.img_size)
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    va_dl = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    te_dl = DataLoader(te_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_unet(base=args.base).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net params: {n_params:,}")

    criterion = BCEDiceLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / "best_stage1_unet.pth"
    best_dice = -1.0
    history: list[dict] = []

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(tr_dl, 1):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            seen += len(x)
            if step % 20 == 0:
                print(f"  ep{ep} {seen}/{len(tr_ds)} loss={total / seen:.4f} "
                      f"({time.time() - t0:.0f}s)")
        scheduler.step()

        va = evaluate(model, va_dl, args.device)
        tr_loss = total / max(seen, 1)
        history.append({"epoch": ep, "train_loss": tr_loss, **{f"val_{k}": v for k, v in va.items()}})
        print(f"Ep{ep:02d}/{args.epochs} loss={tr_loss:.4f} val_dice={va['dice_mean']:.4f} "
              f"iou={va['iou_mean']:.4f} ({time.time() - t0:.0f}s)")

        if va["dice_mean"] > best_dice:
            best_dice = va["dice_mean"]
            torch.save(model.state_dict(), best_path)
            print(f"  -> lưu checkpoint mới (val_dice={best_dice:.4f})")

    print(f"\nBest val Dice = {best_dice:.4f} -> {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test = evaluate(model, te_dl, args.device)
    print("TEST:", json.dumps(test, indent=2))

    metrics = {
        "best_val_dice": best_dice,
        "test": test,
        "history": history,
        "config": {
            "arch": "unet",
            "base": args.base,
            "img_size": args.img_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "loss": "BCEDiceLoss",
            "ground_truth": "CheXmask (left|right, không gồm heart)",
        },
    }
    metrics_path = args.out_dir / "stage1_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {metrics_path}")


if __name__ == "__main__":
    main()
