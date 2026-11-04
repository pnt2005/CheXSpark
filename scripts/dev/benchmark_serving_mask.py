"""Đo chênh lệch đường suy luận (serving) so với đường train trên dev test 344 ảnh.

Lúc **train** Stage-2 dùng mask CheXmask (sinh sẵn cho 112k ảnh NIH). Lúc **suy
luận** trên ảnh mới không có CheXmask → dùng U-Net Stage-1, và model production
(`mask_input=3ch`) cần cả kênh `ảnh×(phổi∪ tim)` nên serving phải ước lượng
vùng tim. Script này trả lời: **mất bao nhiêu mAUROC** cho từng cách ước lượng.

Bốn đường đối chứng (cùng model, cùng calibration, cùng 344 ảnh test dev):

    chexmask  mask CheXmask thật (đường train — tham chiếu)
    none      U-Net phổi, kênh 2 = kênh 1 (không ước lượng tim)
    closing   U-Net phổi + tim ≈ đóng morphology + lấp lỗ
    scanline  U-Net phổi + tim ≈ lấp ngang mỗi hàng giữa hai phổi (mặc định serving)

Usage::

    python scripts/dev/benchmark_serving_mask.py                 # full 344
    python scripts/dev/benchmark_serving_mask.py --limit 64      # thử nhanh
    python scripts/dev/benchmark_serving_mask.py --modes none,scanline
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dl.preprocess import LABEL_COLS, load_gray, lung_mask, build_input  # noqa: E402
from dl.segmentation import predict_mask  # noqa: E402
from serving.pipeline import HEART_FN, InferenceEngine  # noqa: E402

SILVER_MASKED = ROOT / "data/tmp/silver_masked/cxr14_masked"
BRONZE_IMG = ROOT / "data/bronze/images"
OUT_JSON = ROOT / "data/outputs/serving_mask_benchmark.json"

MODES = ("chexmask", "none", "closing", "scanline")


def auc_rank(y: np.ndarray, s: np.ndarray) -> float:
    """AUC theo rank (Mann-Whitney), tự xử lý tie — tránh phụ thuộc sklearn."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # trung bình rank cho các giá trị bằng nhau
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_split(split: str) -> pd.DataFrame:
    """Đọc dev Silver đã join CheXmask — partition theo thư mục `split=<tên>`."""
    part = SILVER_MASKED / f"split={split}"
    files = sorted(part.rglob("*.parquet")) if part.exists() else sorted(
        SILVER_MASKED.rglob("*.parquet")
    )
    if not files:
        raise FileNotFoundError(f"Không thấy Parquet tại {SILVER_MASKED}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["split"] = split  # Spark ghi partition bằng thư mục, không ghi cột
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=0, help="0 = toàn bộ split")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--calibration", default=None, help="path stage2_metrics.json")
    ap.add_argument("--no-registry", action="store_true", help="load .pth local thay vì MLflow")
    ap.add_argument("--json", action="store_true", help=f"ghi kết quả ra {OUT_JSON}")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"Mode không hợp lệ {bad} — chỉ có {MODES}")

    df = load_split(args.split)
    if args.limit:
        df = df.head(args.limit)
    print(f"Dataset: {len(df)} ảnh split={args.split}")

    eng = InferenceEngine(
        use_registry=not args.no_registry,
        calibration_path=args.calibration,
        heart_mode="scanline",  # loop tự đổi mode nên giá trị này không dùng tới
    )
    y = df[LABEL_COLS].to_numpy(dtype=np.float32)

    # Sinh mask 1 lần cho mỗi ảnh (U-Net) + mask CheXmask thật
    lung_unet, masks_ref = [], []
    t0 = time.perf_counter()
    for _, row in df.iterrows():
        path = BRONZE_IMG / row["image_index"]
        gray = load_gray(path)
        lung_unet.append(predict_mask(eng.stage1, gray, device=eng.device))
        lm = lung_mask(row, include_heart=False)
        both = lung_mask(row, include_heart=True)
        masks_ref.append((lm, np.clip(both - lm, 0, 1)))
    print(f"Sinh mask U-Net + CheXmask: {time.perf_counter() - t0:.1f}s")

    # Dice U-Net vs CheXmask (lung) — chất lượng Stage-1 trên chính ảnh này
    dice = [
        2 * float((u * r).sum()) / (u.sum() + r.sum() + 1e-6)
        for u, (r, _) in zip(lung_unet, masks_ref)
    ]
    print(f"Dice U-Net vs CheXmask (lung): mean={np.mean(dice):.4f}")

    results: dict[str, dict] = {}
    probs_by_mode: dict[str, np.ndarray] = {}

    for mode in modes:
        t0 = time.perf_counter()
        probs = np.zeros((len(df), len(LABEL_COLS)), dtype=np.float32)
        for i in range(0, len(df), args.batch):
            sl = slice(i, i + args.batch)
            grays = [load_gray(BRONZE_IMG / n) for n in df["image_index"].iloc[sl]]
            xs, ys = [], []
            for k, gray in enumerate(grays):
                idx = i + k
                if mode == "chexmask":
                    lung, heart = masks_ref[idx]
                else:
                    lung = lung_unet[idx]
                    heart = HEART_FN[mode](lung) if mode != "none" else None
                # build_input trả (1,3,H,W) → gộp batch thủ công
                xs.append(build_input(gray, lung, heart, eng.mask_input))
                ys.append(idx)
            x = torch.cat(xs, dim=0)
            with torch.no_grad():
                logits = eng.stage2(x)
            p = torch.sigmoid(logits / eng.temperature).numpy()
            for k, idx in enumerate(ys):
                probs[idx] = p[k]
        probs_by_mode[mode] = probs
        per = {
            d: auc_rank(y[:, j], probs[:, j]) for j, d in enumerate(LABEL_COLS)
        }
        valid = [v for v in per.values() if not np.isnan(v)]
        results[mode] = {
            "mauroc": float(np.mean(valid)),
            "per_label_auc": per,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        print(f"  [{mode:8s}] mAUROC={results[mode]['mauroc']:.4f}  "
              f"({results[mode]['seconds']}s)")

    # Bảng đối chiếu
    print("\n=== mAUROC theo đường suy luận ===")
    base = results.get("chexmask", {}).get("mauroc")
    for mode in modes:
        m = results[mode]["mauroc"]
        delta = f"  ({m - base:+.4f})" if base and mode != "chexmask" else ""
        print(f"  {mode:10s} {m:.4f}{delta}")

    if len(modes) > 1:
        print("\n=== mAUROC từng bệnh (chexmask → từng mode) ===")
        names = list(results[modes[0]]["per_label_auc"])
        header = f"  {'bệnh':22s}" + "".join(f"{m:>10s}" for m in modes)
        print(header)
        for d in names:
            row = f"  {d:22s}"
            for m in modes:
                v = results[m]["per_label_auc"][d]
                row += f"{'—':>10s}" if np.isnan(v) else f"{v:10.4f}"
            print(row)

    if args.json:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": args.split,
            "n_images": len(df),
            "dice_unet_vs_chexmask": round(float(np.mean(dice)), 4),
            "mask_input": eng.mask_input,
            "results": results,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nĐã ghi {OUT_JSON}")


if __name__ == "__main__":
    main()
