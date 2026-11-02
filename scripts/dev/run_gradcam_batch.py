"""Stage-3: sinh Grad-CAM cho 14 bệnh trên tập test của Stage-2.

Quy trình:
  1. Load checkpoint DenseNet121 Stage-2 → dự đoán xác suất 14 bệnh cho tập test
  2. Với mỗi bệnh: chọn top-K ảnh **dương tính thật** (label=1) có xác suất cao nhất
  3. Sinh Grad-CAM cho đúng lớp bệnh đó → phủ heatmap lên ảnh (chỉ trong vùng phổi)
  4. Ghép contact sheet mỗi bệnh: [ảnh gốc | mask phổi | heatmap | overlay]

Output: data/outputs/gradcam/
  - gradcam_<Disease>.png     contact sheet top-K ca bệnh
  - cam_<image>_<Disease>.png ảnh overlay đơn lẻ
  - gradcam_index.csv         (image_index, disease, prob, label, file)

Usage:
    python scripts/dev/run_gradcam_batch.py --top-k 4
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

from dl.gradcam import GradCAM, jet_rgb, mask_visual, overlay_heatmap, to_pil, upsample_cam  # noqa: E402
from dl.model import load_stage2_model  # noqa: E402
from dl.preprocess import DISEASES, build_tensor, load_gray, lung_mask  # noqa: E402
from dl.segmentation import load_stage1_model, predict_mask  # noqa: E402

PANEL = 224


def predict_all(model, df: pd.DataFrame, images_dir: Path, device: str,
                mask_fn=None, batch_size: int = 32) -> np.ndarray:
    probs = np.zeros((len(df), len(DISEASES)), dtype=np.float32)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        tensors = []
        for _, row in chunk.iterrows():
            gray = load_gray(images_dir / row["image_index"])
            mask = mask_fn(row, gray) if mask_fn else lung_mask(row)
            tensors.append(build_tensor(gray, mask))
        x = torch.cat(tensors).to(device)
        with torch.no_grad():
            probs[start:start + len(chunk)] = torch.sigmoid(model(x)).cpu().numpy()
        if (start // batch_size) % 4 == 0:
            print(f"  predict {min(start + batch_size, len(df))}/{len(df)}")
    return probs


def contact_sheet(rows: list[list[Image.Image]], title: str) -> Image.Image:
    """Ghép lưới ảnh + tiêu đề."""
    if not rows:
        return Image.new("RGB", (PANEL, PANEL), "black")
    w, h = PANEL * len(rows[0]), PANEL * len(rows) + 26
    sheet = Image.new("RGB", (w, h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 7), title, fill=(255, 255, 255))
    for r, row_imgs in enumerate(rows):
        for c, im in enumerate(row_imgs):
            sheet.paste(im, (c * PANEL, 26 + r * PANEL))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "data/kaggle/stage2-nih-dev/stage2_train.csv")
    ap.add_argument("--images-dir", type=Path, default=ROOT / "data/bronze/images")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "notebooks/kaggle/best_stage2_densenet121.pth")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/outputs/gradcam")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--target", default="denseblock3",
                    help="Lớp đích trong model.features (denseblock3=14x14, denseblock4=7x7)")
    ap.add_argument("--diseases", default="",
                    help="Danh sách bệnh cách nhau dấu phẩy (mặc định: cả 14)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--from-registry", action="store_true",
                    help="Nạp model theo alias từ MLflow Model Registry (bỏ qua --ckpt)")
    ap.add_argument("--alias", default="production",
                    help="Alias registry dùng khi có --from-registry")
    ap.add_argument("--mask-source", choices=("chexmask", "unet"), default="chexmask",
                    help="chexmask = mask GT CheXmask; unet = do U-Net Stage-1 tự sinh")
    ap.add_argument("--stage1-ckpt", type=Path,
                    default=ROOT / "notebooks/kaggle/best_stage1_unet.pth")
    ap.add_argument("--stage1-from-registry", action="store_true",
                    help="Nạp U-Net Stage-1 từ MLflow Registry thay vì --stage1-ckpt")
    args = ap.parse_args()

    wanted = [d.strip() for d in args.diseases.split(",") if d.strip()] or DISEASES
    unknown = [d for d in wanted if d not in DISEASES]
    if unknown:
        raise SystemExit(f"Bệnh không hợp lệ: {unknown}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    test = df[df["split"] == args.split].reset_index(drop=True)
    src = f"registry@{args.alias}" if args.from_registry else args.ckpt.name
    print(f"{args.split}: {len(test)} ảnh | model={src} | device={args.device}")

    if args.from_registry:
        from mlops.registry import load_stage2_from_registry  # cần mlflow (venv-mlops)

        model = load_stage2_from_registry(alias=args.alias, device=args.device)
    else:
        model = load_stage2_model(args.ckpt, args.device)

    # Nguồn mask phổi: CheXmask (GT, dùng lúc đánh giá) hoặc U-Net Stage-1 tự sinh
    # (đúng kịch bản triển khai: ảnh mới từ bệnh viện không có CheXmask).
    if args.mask_source == "unet":
        if args.stage1_from_registry:
            from mlops.registry import load_stage1_from_registry

            seg = load_stage1_from_registry(alias=args.alias, device=args.device)
            seg_src = f"stage1@registry:{args.alias}"
        else:
            seg = load_stage1_model(args.stage1_ckpt, args.device)
            seg_src = args.stage1_ckpt.name
        print(f"mask source: U-Net ({seg_src})")

        def mask_fn(row, gray):
            return predict_mask(seg, gray, args.device)
    else:
        print("mask source: CheXmask (ground truth)")

        def mask_fn(row, gray):
            return lung_mask(row)

    target_layer = getattr(model.features, args.target)
    probs = predict_all(model, test, args.images_dir, args.device, mask_fn=mask_fn)

    index_rows = []
    with GradCAM(model, target_layer) as cam_gen:
        for d_idx, disease in enumerate(DISEASES):
            if disease not in wanted:
                continue
            label_col = f"label_{disease}"
            positives = test.index[(test[label_col] == 1)].tolist()
            if not positives:
                print(f"[skip] {disease}: không có ca dương tính trong {args.split}")
                continue

            ranked = sorted(positives, key=lambda i: probs[i, d_idx], reverse=True)
            chosen = ranked[:args.top_k]

            sheet_rows = []
            for i in chosen:
                row = test.iloc[i]
                gray = load_gray(args.images_dir / row["image_index"])
                mask = mask_fn(row, gray)
                x = build_tensor(gray, mask).to(args.device)

                cam = cam_gen.generate(x, d_idx)
                heat_only = jet_rgb(upsample_cam(cam, PANEL))
                over = overlay_heatmap(gray, cam, mask)

                base_rgb = (np.stack([gray] * 3, -1) * 255).astype(np.uint8)
                sheet_rows.append([
                    to_pil(base_rgb),
                    to_pil(mask_visual(gray, mask)),
                    to_pil(heat_only),
                    to_pil(over),
                ])

                single = args.out_dir / f"cam_{row['image_index'].replace('.png', '')}_{disease}.png"
                to_pil(over).save(single)
                index_rows.append((
                    row["image_index"], disease, float(probs[i, d_idx]), 1, single.name
                ))
                print(f"  {disease}: {row['image_index']} p={probs[i, d_idx]:.3f}")

            title = (f"{disease} - top {len(chosen)} true positives (highest probability) | "
                     f"cols: original | lung mask | Grad-CAM | overlay")
            sheet = contact_sheet(sheet_rows, title)
            sheet_path = args.out_dir / f"gradcam_{disease}.png"
            sheet.save(sheet_path)
            print(f"  -> {sheet_path.name}")

    idx = pd.DataFrame(index_rows, columns=["image_index", "disease", "prob", "label", "file"])
    idx.to_csv(args.out_dir / "gradcam_index.csv", index=False)
    print(f"Đã ghi {len(index_rows)} heatmap + {args.out_dir / 'gradcam_index.csv'}")


if __name__ == "__main__":
    main()
