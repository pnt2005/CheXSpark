# -*- coding: utf-8 -*-
"""Đẩy bài đánh giá Stage-2-theo-mask (`eval_stage2_masks.py`) lên Kaggle chạy GPU.

Script kernel (kernel_type=script) — không cần upload dataset mới:

    1. Dataset mount: ảnh 224 (khanfashee) + CSV 112k + mask pack 112k (GT)
    2. Kernel output mount: `chexspark-stage2-densenet121` (checkpoint Stage-2 v3)
       + `chexspark-stage1-unet` (checkpoint Stage-1 v2) → làm weights, không tải gì
    3. Push → poll → tải `stage2_mask_eval.json` về `data/outputs/`

Usage:
    python scripts/kaggle/eval_masks_on_kaggle.py --check      # credential trước
    python scripts/kaggle/eval_masks_on_kaggle.py --dry-run    # xem kernel folder
    python scripts/kaggle/eval_masks_on_kaggle.py              # push + chờ + tải (~15')
    python scripts/kaggle/eval_masks_on_kaggle.py --no-wait    # push rồi thoát
    python scripts/kaggle/eval_masks_on_kaggle.py --wait-only  # chờ run đang chạy

    # Tuỳ chọn patch CONFIG trong script: --split test --modes chexmask,scanline,none
    #                                     --limit 0 --batch 64
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_on_kaggle as tok  # noqa: E402  (reuse CLI wrapper + push/poll/fetch)

ROOT = tok.ROOT
EVAL_SRC = HERE / "eval_stage2_masks.py"
KERNEL_SLUG = "chexspark-stage2-mask-eval"
TITLE = "CheXSpark Stage2 Mask Eval"        # slugify(TITLE) == KERNEL_SLUG
KERNEL_SOURCES = ["chexspark-stage2-densenet121", "chexspark-stage1-unet"]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_dir(username: str, overrides: dict) -> Path:
    assert slugify(TITLE) == KERNEL_SLUG, "title/slug lệch -> Kaggle 409"
    out = ROOT / "data/kaggle/kernels/eval"
    out.mkdir(parents=True, exist_ok=True)

    src = EVAL_SRC.read_text(encoding="utf-8")
    changed = []
    for key, value in overrides.items():
        # các hằng CONFIG trong EVAL_SRC (SPLIT/MODES/LIMIT/BATCH)
        pattern = re.compile(rf"^({key}\s*=\s*)([^#\n]*)(#[^\n]*)?$", re.MULTILINE)

        def _sub(m: re.Match, v: str = value) -> str:
            comment = f" {m.group(3)}" if m.group(3) else ""
            return f"{m.group(1)}{v}{comment}"

        src, n = pattern.subn(_sub, src)
        if n:
            changed.append(f"{key}={value}")
    (out / EVAL_SRC.name).write_text(src, encoding="utf-8")

    meta = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": TITLE,
        "code_file": EVAL_SRC.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            tok.IMAGE_DATASET,
            f"{username}/chexspark-stage2-nih-112k",
            f"{username}/chexspark-masks-224-112k",
        ],
        "kernel_sources": [f"{username}/{s}" for s in KERNEL_SOURCES],
        "competition_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKernel folder: {out}")
    print(f"  id           : {meta['id']} (script)")
    print(f"  datasets     : {', '.join(meta['dataset_sources'])}")
    print(f"  kernel src   : {', '.join(meta['kernel_sources'])} (weights)")
    if changed:
        print(f"  config patch : {', '.join(changed)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra credential")
    ap.add_argument("--dry-run", action="store_true", help="chỉ build kernel folder")
    ap.add_argument("--username", default=None)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--modes", default="chexmask,scanline,none")
    ap.add_argument("--limit", type=int, default=0, help="0 = toàn bộ split")
    ap.add_argument("--batch", type=int, default=0, help="0 = tự chọn (64 GPU/16 CPU)")
    ap.add_argument("--run-timeout", type=int, default=3600)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--wait-timeout", type=int, default=7200)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--wait-only", action="store_true",
                    help="không push — chờ kernel ĐANG chạy rồi tải kết quả")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "data/kaggle/output/eval")
    ap.add_argument("--print", dest="do_print", action="store_true", default=True,
                    help="in nội dung stage2_mask_eval.json sau khi tải")
    args = ap.parse_args()

    kaggle = tok.kaggle_bin()
    username = args.username or tok.detect_username(kaggle)
    if not username:
        raise SystemExit("Không dò được username Kaggle — truyền --username.")

    if args.check:
        proc = tok.run([kaggle, "datasets", "list", "-m", "--page-size", "1"], check=False)
        if proc.returncode != 0:
            raise SystemExit("Chưa xác thực Kaggle — chạy `kaggle auth login`.")
        print(f"Xác thực OK | username: {username}")
        print(f"weights mount: {', '.join(KERNEL_SOURCES)}")
        return

    kernel_id = f"{username}/{KERNEL_SLUG}"
    if not args.wait_only:
        build_dir(username, {
            "SPLIT": f"'{args.split}'",
            "MODES": f"'{args.modes}'",
            "LIMIT": str(args.limit),
            "BATCH": str(args.batch),
        })
        if args.dry_run:
            print("\n[dry-run] không push.")
            return
        status, kernel_id = tok.push_and_wait(
            kaggle, kernel_id, ROOT / "data/kaggle/kernels/eval",
            accelerator=None, run_timeout=args.run_timeout, poll=args.poll,
            wait_timeout=args.wait_timeout, no_wait=args.no_wait)
        if status == "queued":
            return
    else:
        status, kernel_id = tok.wait_for_kernel(
            kaggle, kernel_id, poll=args.poll, wait_timeout=args.wait_timeout)

    if status != "complete":
        try:
            tok.fetch_output(kaggle, kernel_id, args.output_dir)
        except SystemExit:
            pass
        raise SystemExit(f"Kernel '{status}'. Xem log: {kaggle} kernels logs {kernel_id}")

    tok.fetch_output(kaggle, kernel_id, args.output_dir)
    hits = list(args.output_dir.rglob("stage2_mask_eval.json"))
    if not hits:
        raise SystemExit("Tải xong nhưng không thấy stage2_mask_eval.json.")
    dest = ROOT / "data/outputs/stage2_mask_eval.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hits[0], dest)
    data = json.loads(dest.read_text(encoding="utf-8"))
    print(f"\n=== Kết quả (split={data['split']}, n={data['n']}) ===")
    for mode, r in data["modes"].items():
        delta = data["delta_vs_chexmask"].get(mode)
        suffix = "" if mode == "chexmask" else f"  ({delta:+.4f})"
        print(f"  {mode:10s} mAUROC {r['mauroc']:.4f}{suffix}")
    print(f"Đã ghi {dest}")


if __name__ == "__main__":
    main()
