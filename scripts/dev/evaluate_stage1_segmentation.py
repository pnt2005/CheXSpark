"""Đánh giá Stage-1 (U-Net) trên tập test so với mask CheXmask (ground truth).

Chỉ số: Dice + IoU từng ảnh, phân bố (mean/median/p10), tỉ lệ ảnh đạt Dice ≥ 0.7
(ngưỡng chất lượng mask khuyến nghị của Silver layer). Xuất CSV để vẽ biểu đồ +
contact sheet các ca tệ nhất (để soi lỗi: ảnh nghiêng, cắt phổi, nhiễu...).

Usage:
    python scripts/dev/evaluate_stage1_segmentation.py
    python scripts/dev/evaluate_stage1_segmentation.py --split val --viz 6
    python scripts/dev/evaluate_stage1_segmentation.py --from-registry --alias production
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dl.preprocess import IMG_SIZE, decode_rle  # noqa: E402
from dl.segmentation import dice_coef, iou_score, load_stage1_model, load_gray, predict_mask  # noqa: E402

PANEL = 224


def gt_mask(row, img_size: int = IMG_SIZE) -> np.ndarray:
    """Mask GT: phổi trái | phải (CheXmask), resize NEAREST."""
    h, w = int(row["mask_height"]), int(row["mask_width"])
    m = decode_rle(row["mask_left_rle"], h, w) | decode_rle(row["mask_right_rle"], h, w)
    m = Image.fromarray((m * 255).astype(np.uint8)).resize((img_size, img_size), Image.NEAREST)
    return (np.array(m) > 127).astype(np.float32)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def build_sheet(items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], title: str) -> Image.Image:
    """Lưới [ảnh | GT | dự đoán | overlay] cho các ca tệ nhất."""
    if not items:
        return Image.new("RGB", (PANEL, PANEL), "black")
    w, h = PANEL * 4, PANEL * len(items) + 26
    sheet = Image.new("RGB", (w, h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 7), title, fill=(255, 255, 255))
    for r, (_, gray, gt, pred) in enumerate(items):
        overlay = np.stack([gray] * 3, -1).copy()
        overlay[..., 1] = np.maximum(overlay[..., 1], pred * 0.9)   # xanh = dự đoán
        overlay[..., 0] = np.maximum(overlay[..., 0], gt * 0.6)     # đỏ = GT
        for c, im in enumerate([np.stack([gray] * 3, -1), np.stack([gt] * 3, -1),
                                np.stack([pred] * 3, -1), overlay]):
            sheet.paste(to_pil(im), (c * PANEL, 26 + r * PANEL))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "data/kaggle/stage2-nih-dev/stage2_train.csv")
    ap.add_argument("--images-dir", type=Path, default=ROOT / "data/bronze/images")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "notebooks/kaggle/best_stage1_unet.pth")
    ap.add_argument("--from-registry", action="store_true",
                    help="nạp model theo alias MLflow Registry (cần venv-mlops)")
    ap.add_argument("--alias", default="production")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/outputs/stage1")
    ap.add_argument("--viz", type=int, default=4, help="số ca tệ nhất vẽ vào contact sheet")
    args = ap.parse_args()

    if args.from_registry:
        from mlops.registry import load_stage1_from_registry

        model = load_stage1_from_registry(alias=args.alias, device=args.device)
        src = f"registry@{args.alias}"
    else:
        model = load_stage1_model(args.ckpt, args.device)
        src = args.ckpt.name

    df = pd.read_csv(args.csv, dtype={"image_index": str})
    df = df[(df.split == args.split) & df["mask_left_rle"].notna()].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"{args.split}: {len(df)} ảnh | model={src} | device={args.device}")

    rows = []
    for i, row in df.iterrows():
        gray = load_gray(args.images_dir / row["image_index"])
        gt = gt_mask(row)
        pred = predict_mask(model, gray, args.device)
        rows.append({
            "image_index": row["image_index"],
            "dice": dice_coef(pred, gt),
            "iou": iou_score(pred, gt),
            "gt_fg_frac": float(gt.mean()),
            "pred_fg_frac": float(pred.mean()),
        })

    res = pd.DataFrame(rows)
    dice = res["dice"].to_numpy()
    summary = {
        "n": len(res),
        "dice_mean": float(dice.mean()),
        "dice_median": float(np.median(dice)),
        "dice_p10": float(np.percentile(dice, 10)),
        "dice_min": float(dice.min()),
        "iou_mean": float(res["iou"].mean()),
        "frac_dice_ge_07": float((dice >= 0.7).mean()),
        "frac_dice_ge_09": float((dice >= 0.9).mean()),
    }

    print("\n=== KẾT QUẢ ===")
    for k, v in summary.items():
        print(f"  {k:<17}: {v:.4f}" if isinstance(v, float) else f"  {k:<17}: {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.out_dir / f"stage1_eval_{args.split}.csv"
    res.sort_values("dice").to_csv(csv_out, index=False)
    print(f"\nĐã ghi {csv_out}")

    worst = res.sort_values("dice").head(args.viz)
    for _, r in worst.iterrows():
        print(f"  ca tệ: {r['image_index']} dice={r['dice']:.3f} "
              f"gt={r['gt_fg_frac']:.2f} pred={r['pred_fg_frac']:.2f}")

    if args.viz:
        items = []
        for _, r in worst.iterrows():
            row = df[df.image_index == r["image_index"]].iloc[0]
            gray = load_gray(args.images_dir / row["image_index"])
            items.append((r["image_index"], gray, gt_mask(row), predict_mask(model, gray, args.device)))
        sheet = build_sheet(items, f"Stage-1 {args.split} - {len(items)} ca Dice thấp nhất "
                                   f"| cols: anh | GT CheXmask | du doan | overlay (do=GT, xanh=pred)")
        sheet_path = args.out_dir / f"stage1_worst_{args.split}.png"
        sheet.save(sheet_path)
        print(f"Đã ghi {sheet_path}")


if __name__ == "__main__":
    main()
