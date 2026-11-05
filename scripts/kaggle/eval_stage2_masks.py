# -*- coding: utf-8 -*-
"""Đánh giá Stage-2 trên tập test với 3 đường mask đầu vào — chexmask / scanline / none.

Lúc **train** Stage-2 dùng mask CheXmask (GT). Lúc **suy luận** phải dùng U-Net
Stage-1 + ước lượng tim (proxy). Script này đo mAUROC 14 bệnh của từng đường trên
cùng tập test, cùng checkpoint — trả lời "mất bao nhiêu khi đưa vào serving".

    chexmask  GT CheXmask (lungs + heart từ mask pack) — đường train, tham chiếu
    scanline  U-Net v2 + tim ước lượng scanline (mặc định serving)
    none      U-Net v2, không ước lượng tim (kênh 2 = kênh 1)

Chạy được 2 nơi (tự dò):
  - **Local** (mặc định): dev test 344, checkpoint `notebooks/kaggle/*.pth`
        python scripts/kaggle/eval_stage2_masks.py
  - **Kaggle**: script kernel — mount dataset ảnh/CSV/mask pack + output kernel
    Stage-1/Stage-2 làm weights (đẩy qua `eval_masks_on_kaggle.py`).

Đầu ra: JSON `{mAUROC từng đường, per-label, Δ so với đường train}`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

ROOT = Path(__file__).resolve().parents[2]
ON_KAGGLE = os.path.isdir("/kaggle/input")

# ==== CONFIG (wrapper eval_masks_on_kaggle.py patch các dòng này khi push) ====
SPLIT = "test"
MODES = "chexmask,scanline,none"
LIMIT = 0        # 0 = toàn bộ split
BATCH = 0        # 0 = tự chọn (64 nếu có CUDA, ngược lại 16)
# =============================== end CONFIG ===================================

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Path: Kaggle mount hoặc local repo
# --------------------------------------------------------------------------- #
def resolve_input(ref: str) -> str:
    """Tìm thư mục dataset/kernel-output đã mount trên Kaggle."""
    if not ref:
        return ""
    slug = ref.split("/")[-1]
    for cand in (f"/kaggle/input/datasets/{ref}", f"/kaggle/input/{ref}",
                 f"/kaggle/input/{slug}"):
        if os.path.isdir(cand):
            return cand
    hits = [h for h in sorted(glob.glob(f"/kaggle/input/**/{slug}", recursive=True))
            if os.path.isdir(h)]
    return hits[0] if hits else ""


def find_file(name: str) -> str:
    """Dò 1 file đã mount (kernel output) hoặc trong repo local."""
    if ON_KAGGLE:
        hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if hits:
            return max(hits, key=os.path.getmtime)
    return ""


def default_paths() -> dict:
    if ON_KAGGLE:
        p = {
            "img_root": resolve_input("khanfashee/nih-chest-x-ray-14-224x224-resized"),
            "meta_csv": os.path.join(resolve_input("pnt2005/chexspark-stage2-nih-112k"),
                                     "stage2_train.csv"),
            "mask_dir": resolve_input("pnt2005/chexspark-masks-224-112k"),
            "stage1_ckpt": find_file("best_stage1_unet.pth"),
            "stage2_ckpt": find_file("best_stage2_densenet121.pth"),
            "metrics_json": find_file("stage2_metrics.json"),
        }
    else:
        p = {
            "img_root": str(ROOT / "data/bronze/images"),
            "meta_csv": str(ROOT / "data/kaggle/stage2-nih-dev/stage2_train.csv"),
            "mask_dir": str(ROOT / "data/kaggle/stage2-masks-224-112k"),
            "stage1_ckpt": str(ROOT / "notebooks/kaggle/best_stage1_unet.pth"),
            "stage2_ckpt": str(ROOT / "notebooks/kaggle/best_stage2_densenet121.pth"),
            "metrics_json": str(ROOT / "notebooks/kaggle/stage2_metrics.json"),
        }
    return p


# --------------------------------------------------------------------------- #
# Model (copy y hệt kiến trúc train/serving — kernel không có src/)
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net 4 tầng xuống — base tự suy ra từ checkpoint (dev=16, production=32)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        b = base
        self.enc1, self.enc2 = ConvBlock(in_ch, b), ConvBlock(b, b * 2)
        self.enc3, self.enc4 = ConvBlock(b * 2, b * 4), ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 8, b * 16)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2); self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2); self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2); self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2); self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def load_unet(path: str) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    base = int(state["enc1.net.0.weight"].shape[0])
    m = UNet(base=base)
    m.load_state_dict(state)
    m.eval()
    print(f"[stage1] {Path(path).name} | base={base} | "
          f"{sum(p.numel() for p in m.parameters()):,} params", flush=True)
    return m


def load_densenet(path: str, n_classes: int) -> nn.Module:
    m = models.densenet121(weights=None)          # weights=None: state_dict ckpt tự load đủ
    m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    print(f"[stage2] {Path(path).name} | {sum(p.numel() for p in m.parameters()):,} params",
          flush=True)
    return m


def heart_scanline(lung: np.ndarray) -> np.ndarray:
    """Lấp ngang mỗi hàng: từ cột phổi trái nhất → phải nhất (trùng serving)."""
    m = np.asarray(lung) > 0.5
    out = np.zeros_like(m, dtype=np.float32)
    for y in range(m.shape[0]):
        xs = np.flatnonzero(m[y])
        if xs.size:
            out[y, xs[0]: xs[-1] + 1] = 1.0
    return out


def build_batch(g, lung, heart, mask_input: str) -> torch.Tensor:
    """(B,1,H,W) + mask → input Stage-2 (B,3,H,W) đã normalize — khớp build_input."""
    if mask_input == "none" or lung is None:
        x = torch.cat([g] * 3, dim=1)
    elif mask_input == "masked":
        x = torch.cat([g * lung] * 3, dim=1)
    else:                                        # "3ch" (model production)
        hm = heart if heart is not None else torch.zeros_like(lung)
        x = torch.cat([g, g * lung, g * torch.clamp(lung + hm, 0, 1)], dim=1)
    mean = torch.tensor(MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------- #
# Dataset: ảnh xám + GT pack (lungs|heart) + nhãn
# --------------------------------------------------------------------------- #
class EvalDS(Dataset):
    def __init__(self, frame: pd.DataFrame, mask_pack, mask_heart_pack, mask_pos,
                 labels: list[str]):
        self.df = frame.reset_index(drop=True)
        self.mask_pack = mask_pack
        self.mask_heart_pack = mask_heart_pack
        self.mask_pos = mask_pos
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        g = np.array(img).astype(np.float32) / 255.0
        key = (row["split"], row["image_index"])
        pos = self.mask_pos[key]
        lung = np.unpackbits(self.mask_pack[row["split"]][pos]).reshape(IMG_SIZE, IMG_SIZE)
        if self.mask_heart_pack is not None:
            heart = np.unpackbits(self.mask_heart_pack[row["split"]][pos]).reshape(
                IMG_SIZE, IMG_SIZE)
        else:
            heart = np.zeros_like(lung)
        y = np.array([row[c] for c in self.labels], dtype=np.float32)
        return (torch.from_numpy(g)[None], torch.from_numpy(lung.astype(np.float32))[None],
                torch.from_numpy(heart.astype(np.float32))[None], torch.from_numpy(y))


def auc_summary(Y: np.ndarray, Z: np.ndarray) -> tuple[float, dict]:
    P = 1.0 / (1.0 + np.exp(-Z))
    per, aucs = {}, []
    for i in range(Y.shape[1]):
        if 0 < Y[:, i].sum() < len(Y):
            a = float(roc_auc_score(Y[:, i], P[:, i]))
            aucs.append(a)
            per[i] = a
        else:
            per[i] = None                 # nhãn degenerate trong tập này
    return float(np.nanmean(aucs)), per


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=SPLIT, choices=("train", "val", "test"))
    ap.add_argument("--modes", default=MODES, help="chexmask,scanline,none")
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    for key in ("img_root", "meta_csv", "mask_dir", "stage1_ckpt", "stage2_ckpt",
                "metrics_json"):
        ap.add_argument(f"--{key.replace('_', '-')}", default=None)
    args = ap.parse_args()

    dp = default_paths()
    for key in ("img_root", "meta_csv", "mask_dir", "stage1_ckpt", "stage2_ckpt",
                "metrics_json"):
        if getattr(args, key):
            dp[key] = getattr(args, key)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = args.batch or (64 if device.type == "cuda" else 16)
    workers = args.workers if args.workers is not None else (
        4 if ON_KAGGLE else (0 if os.name == "nt" else 2))
    print(f"device={device} | batch={batch} | split={args.split} | modes={modes} | "
          f"limit={args.limit or 'all'}", flush=True)
    for k, v in dp.items():
        print(f"  {k:14s}= {v or '(thiếu!)'}", flush=True)
    missing = [k for k, v in dp.items() if not v or not os.path.exists(v)]
    if missing:
        raise SystemExit(f"Thiếu path: {missing}")

    # metrics config của model production (mask_input/mask_mode)
    metrics = json.loads(Path(dp["metrics_json"]).read_text(encoding="utf-8"))
    mcfg = metrics.get("config", metrics)
    mask_input = mcfg.get("mask_input", "3ch")
    mask_mode = mcfg.get("mask_mode", "lungs_heart")
    print(f"model config: mask_input={mask_input} mask_mode={mask_mode}", flush=True)
    if mask_input != "3ch":
        print(f"[WARN] mask_input={mask_input} — 3 đường mode sẽ theo config này", flush=True)

    # ---- index ảnh + label + split ----
    img_index = {}
    for dirpath, _, filenames in os.walk(dp["img_root"]):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                img_index[fn] = os.path.join(dirpath, fn)
    print(f"ảnh: {len(img_index)} PNG", flush=True)

    df = pd.read_csv(dp["meta_csv"], dtype={"image_index": str})
    labels = [c for c in df.columns if c.startswith("label_")]
    df["path"] = df["image_index"].map(img_index)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df = df[df.split == args.split].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"rows: {len(df)} | labels: {len(labels)}", flush=True)

    # ---- mask pack GT ----
    mi = pd.read_csv(os.path.join(dp["mask_dir"], "mask_index.csv"),
                     dtype={"image_index": str})
    mask_pos = {(r.split, r.image_index): int(r.mask_row) for r in mi.itertuples(index=False)}
    mask_pack = {s: np.load(os.path.join(dp["mask_dir"], f"masks_{s}.npy"), mmap_mode="r")
                 for s in ("train", "val", "test")}
    hp = os.path.join(dp["mask_dir"], "masks_heart_train.npy")
    mask_heart_pack = ({s: np.load(os.path.join(dp["mask_dir"], f"masks_heart_{s}.npy"),
                                   mmap_mode="r") for s in ("train", "val", "test")}
                       if os.path.exists(hp) else None)
    miss = sum(1 for s, i in zip(df.split, df.image_index) if (s, i) not in mask_pos)
    if miss:
        raise SystemExit(f"mask pack thiếu {miss}/{len(df)} mask")
    gt_heart = mask_mode == "lungs_heart" and mask_heart_pack is not None

    # ---- model ----
    unet = load_unet(dp["stage1_ckpt"]).to(device)
    model = load_densenet(dp["stage2_ckpt"], len(labels)).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ds = EvalDS(df, mask_pack, mask_heart_pack if gt_heart else None, mask_pos, labels)
    dl = DataLoader(ds, batch, shuffle=False, num_workers=workers, pin_memory=True)

    @torch.no_grad()
    def run_mode(mode: str) -> tuple[np.ndarray, np.ndarray]:
        Ys, Zs = [], []
        t0 = time.time()
        for bi, (g, gt_lung, gt_heart_b, y) in enumerate(dl):
            if mode == "chexmask":
                lung, heart = gt_lung, gt_heart_b
            else:
                gd = g.to(device, non_blocking=True)       # U-Net đang ở CUDA
                lung = (torch.sigmoid(unet((gd - 0.5) / 0.5)) > 0.5).float().cpu()
                if mode == "scanline":
                    heart = torch.from_numpy(np.stack([
                        heart_scanline(lung[i, 0].numpy()) for i in range(lung.shape[0])
                    ]))[:, None]
                else:                              # none: không ước lượng tim
                    heart = torch.zeros_like(lung)
            x = build_batch(g, lung, heart, mask_input).to(device, non_blocking=True)
            z = model(x)
            Ys.append(y.numpy())
            Zs.append(z.float().cpu().numpy())
            if bi % 20 == 0:
                print(f"  [{mode}] batch {bi} ({time.time() - t0:.0f}s)", flush=True)
        return np.concatenate(Ys), np.concatenate(Zs)

    results, timing = {}, {}
    for mode in modes:
        print(f"== {mode} ==", flush=True)
        t0 = time.time()
        Y, Z = run_mode(mode)
        mroc, per = auc_summary(Y, Z)
        timing[mode] = round(time.time() - t0, 1)
        results[mode] = {"mauroc": mroc, "per_label": per}
        print(f"  mAUROC {mode} = {mroc:.4f} ({timing[mode]}s)", flush=True)

    labels_name = [c.replace("label_", "") for c in labels]
    base = results.get("chexmask", {}).get("mauroc")
    out = {
        "split": args.split,
        "n": len(df),
        "device": str(device),
        "mask_input": mask_input,
        "mask_mode": mask_mode,
        "batch": batch,
        "labels": labels_name,
        "modes": results,
        "delta_vs_chexmask": {
            m: round(results[m]["mauroc"] - base, 4) for m in results
            if base is not None and m != "chexmask"
        },
        "per_label_delta": {
            m: {labels_name[i]: (None if results["chexmask"]["per_label"][i] is None
                                 or results[m]["per_label"][i] is None
                                 else round(results[m]["per_label"][i]
                                            - results["chexmask"]["per_label"][i], 4))
                for i in range(len(labels))}
            for m in results if m != "chexmask"
        },
        "timing_s": timing,
    }

    print("\n=== mAUROC theo đường mask ===", flush=True)
    for m in results:
        delta = out["delta_vs_chexmask"].get(m)
        suffix = "" if m == "chexmask" else f"  ({delta:+.4f})"
        print(f"  {m:10s} {results[m]['mauroc']:.4f}{suffix}", flush=True)

    out_path = Path(args.out or (ROOT / "data/outputs/stage2_mask_eval.json"))
    if ON_KAGGLE:
        out_path = Path("stage2_mask_eval.json")     # kernel output = cwd
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {out_path}", flush=True)


if __name__ == "__main__":
    main()
