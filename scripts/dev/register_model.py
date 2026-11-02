"""Đăng ký model Stage-1 / Stage-2 vào MLflow Model Registry rồi kiểm chứng.

Việc script làm (giống nhau cho cả 2 stage):
  1. Log run mới: params (kiến trúc, dataset, cấu hình train, sha256 checkpoint),
     metrics (lấy từ `stage1_metrics.json` / `stage2_metrics.json`)
  2. `mlflow.pytorch.log_model` + signature, đăng ký thành model version mới
  3. Gán alias `production` cho version đó (promote)
  4. **Kiểm chứng**: nạp lại model từ registry, so sánh logits với checkpoint local
     trên cùng một input ngẫu nhiên

Usage:
    # Stage-2 (phân loại 14 bệnh)
    python scripts/dev/register_model.py --stage 2

    # Stage-1 (U-Net phân đoạn phổi)
    python scripts/dev/register_model.py --stage 1

    # Promote/rollback và xem danh sách version
    python scripts/dev/register_model.py --stage 1 --list
    python scripts/dev/register_model.py --stage 1 --promote 3

Biến môi trường: chỉ cần `MLFLOW_TRACKING_URI` (server chạy chế độ artifact proxy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dl.model import load_stage2_model  # noqa: E402
from dl.segmentation import load_stage1_model  # noqa: E402
from mlops.registry import (  # noqa: E402
    EXPERIMENT,
    MODEL_NAME,
    PRODUCTION_ALIAS,
    STAGE1_EXPERIMENT,
    STAGE1_MODEL_NAME,
    describe,
    load_stage1_from_registry,
    load_stage2_from_registry,
    promote,
    register_stage1_model,
    register_stage2_model,
)

STAGES = {
    1: {
        "register": register_stage1_model,
        "load_local": load_stage1_model,
        "load_registry": load_stage1_from_registry,
        "model_name": STAGE1_MODEL_NAME,
        "experiment": STAGE1_EXPERIMENT,
        "ckpt": ROOT / "notebooks/kaggle/best_stage1_unet.pth",
        "metrics": ROOT / "notebooks/kaggle/stage1_metrics.json",
        "input_shape": (2, 1, 224, 224),
        "primary": "test_dice_mean",
        "label": "test Dice",
    },
    2: {
        "register": register_stage2_model,
        "load_local": load_stage2_model,
        "load_registry": load_stage2_from_registry,
        "model_name": MODEL_NAME,
        "experiment": EXPERIMENT,
        "ckpt": ROOT / "notebooks/kaggle/best_stage2_densenet121.pth",
        "metrics": ROOT / "notebooks/kaggle/stage2_metrics.json",
        "input_shape": (2, 3, 224, 224),
        "primary": "test_mauroc",
        "label": "test mAUROC",
    },
}


def _print_versions(model_name: str, primary: str, label: str) -> None:
    rows = describe(model_name)
    if not rows:
        print(f"(chưa có version nào cho {model_name})")
        return
    print(f"\n{'version':>7} | {'aliases':<20} | {label:>12} | run_id")
    print("-" * 78)
    for r in rows:
        value = r["metrics"].get(primary)
        shown = "-" if value is None else f"{value:.4f}"
        aliases = ",".join(r["aliases"]) or "-"
        print(f"{r['version']:>7} | {aliases:<20} | {shown:>12} | {r['run_id']}")


def _verify(stage_cfg: dict, ckpt: Path, model_name: str, alias: str) -> bool:
    """Nạp model từ registry và so sánh logits với checkpoint local."""
    local = stage_cfg["load_local"](ckpt)
    remote = stage_cfg["load_registry"](model_name, alias)

    torch.manual_seed(0)
    x = torch.randn(*stage_cfg["input_shape"])
    with torch.no_grad():
        out_local = local(x)
        out_remote = remote(x)

    diff = (out_local - out_remote).abs().max().item()
    ok = diff < 1e-5
    print(f"\n[verify] models:/{model_name}@{alias} vs {ckpt.name}")
    print(f"         output shape {tuple(out_local.shape)} | "
          f"max |logit_local - logit_registry| = {diff:.3e}  -> {'OK' if ok else 'LỆCH!'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=2,
                    help="1 = U-Net phân đoạn, 2 = DenseNet121 phân loại")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--alias", default=PRODUCTION_ALIAS)
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--no-register", action="store_true", help="chỉ log run, không tạo model version")
    ap.add_argument("--no-verify", action="store_true", help="bỏ bước so sánh với registry")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê version rồi thoát")
    ap.add_argument("--promote", metavar="VERSION", help="chỉ gán alias cho version có sẵn")
    args = ap.parse_args()

    cfg = STAGES[args.stage]
    model_name = args.model_name or cfg["model_name"]

    if args.list:
        _print_versions(model_name, cfg["primary"], cfg["label"])
        return

    if args.promote:
        uri = promote(model_name, args.promote, args.alias)
        print(f"Đã trỏ alias '{args.alias}' -> version {args.promote} ({uri})")
        _print_versions(model_name, cfg["primary"], cfg["label"])
        return

    ckpt = args.ckpt or cfg["ckpt"]
    result = cfg["register"](
        ckpt=ckpt,
        metrics_json=args.metrics or cfg["metrics"],
        model_name=model_name,
        experiment=args.experiment or cfg["experiment"],
        alias=args.alias,
        tracking_uri=args.tracking_uri,
        register=not args.no_register,
    )

    print(f"\n=== ĐĂNG KÝ MODEL (Stage-{args.stage}) ===")
    print(f"run_id     : {result['run_id']}")
    print(f"model_uri  : {result['model_uri']}")
    if result["version"]:
        print(f"version    : {result['version']}  (alias: {result['alias']})")
    print(f"metrics    : {len(result['metrics'])} chỉ số")
    for key, value in result["metrics"].items():
        if key.startswith(("test_", "best_val_")):
            print(f"  {key:<20}: {value:.4f}")

    _print_versions(model_name, cfg["primary"], cfg["label"])
    print(f"\nMLflow UI: {args.tracking_uri or 'http://localhost:5000'}/#/models/{model_name}/versions")

    if result["version"] and not args.no_verify:
        if not _verify(cfg, ckpt, model_name, result["alias"]):
            sys.exit(1)


if __name__ == "__main__":
    main()
