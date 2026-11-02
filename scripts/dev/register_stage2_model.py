"""Đăng ký model Stage-2 (DenseNet121) vào MLflow Model Registry rồi kiểm chứng.

Việc script làm:
  1. Log run mới vào experiment `chexspark-stage2-registry`:
     params (kiến trúc, dataset, cấu hình train, sha256 checkpoint),
     metrics (val/test mAUROC + AUC từng bệnh lấy từ `stage2_metrics.json`)
  2. `mlflow.pytorch.log_model` + signature (1,3,224,224) → (1,14), đăng ký thành
     model version mới của `chexspark-stage2-densenet121`
  3. Gán alias `production` cho version đó (promote)
  4. **Kiểm chứng**: nạp lại model từ registry bằng `models:/...@production`,
     so sánh output với checkpoint local trên cùng một input ngẫu nhiên

Usage:
    # Đăng ký + promote + verify
    python scripts/dev/register_stage2_model.py

    # Chỉ promote một version đã có lên production (rollback/promote tay)
    python scripts/dev/register_stage2_model.py --promote 3

    # Xem các version hiện có
    python scripts/dev/register_stage2_model.py --list

Biến môi trường (không hard-code credentials):
    MLFLOW_TRACKING_URI      mặc định http://localhost:5000
    MLFLOW_S3_ENDPOINT_URL   http://localhost:9000 (để client upload artifact lên MinIO)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dl.model import load_stage2_model  # noqa: E402
from mlops.registry import (  # noqa: E402
    EXPERIMENT,
    MODEL_NAME,
    PRODUCTION_ALIAS,
    describe,
    load_stage2_from_registry,
    promote,
    register_stage2_model,
)

DEFAULT_CKPT = ROOT / "notebooks/kaggle/best_stage2_densenet121.pth"
DEFAULT_METRICS = ROOT / "notebooks/kaggle/stage2_metrics.json"


def _print_versions(model_name: str) -> None:
    rows = describe(model_name)
    if not rows:
        print(f"(chưa có version nào cho {model_name})")
        return
    print(f"\n{'version':>7} | {'aliases':<20} | {'test mAUROC':>11} | {'val mAUROC':>10} | run_id")
    print("-" * 92)
    for r in rows:
        test = "-" if r["test_mauroc"] is None else f"{r['test_mauroc']:.4f}"
        val = "-" if r["best_val_mauroc"] is None else f"{r['best_val_mauroc']:.4f}"
        aliases = ",".join(r["aliases"]) or "-"
        print(f"{r['version']:>7} | {aliases:<20} | {test:>11} | {val:>10} | {r['run_id']}")


def _verify(ckpt: Path, model_name: str, alias: str) -> bool:
    """Nạp model từ registry và so sánh với checkpoint local."""
    local = load_stage2_model(ckpt)
    remote = load_stage2_from_registry(model_name, alias)

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_local = local(x)
        out_remote = remote(x)

    diff = (out_local - out_remote).abs().max().item()
    ok = diff < 1e-5
    print(f"\n[verify] models:/{model_name}@{alias} vs {ckpt.name}")
    print(f"         max |logit_local - logit_registry| = {diff:.3e}  -> {'OK' if ok else 'LỆCH!'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="checkpoint .pth (state_dict)")
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="stage2_metrics.json")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--alias", default=PRODUCTION_ALIAS)
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--no-register", action="store_true", help="chỉ log run, không tạo model version")
    ap.add_argument("--no-verify", action="store_true", help="bỏ bước so sánh với registry")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê version rồi thoát")
    ap.add_argument("--promote", metavar="VERSION", help="chỉ gán alias cho version có sẵn")
    args = ap.parse_args()

    if args.list:
        _print_versions(args.model_name)
        return

    if args.promote:
        uri = promote(args.model_name, args.promote, args.alias)
        print(f"Đã trỏ alias '{args.alias}' -> version {args.promote} ({uri})")
        _print_versions(args.model_name)
        return

    result = register_stage2_model(
        ckpt=args.ckpt,
        metrics_json=args.metrics,
        model_name=args.model_name,
        experiment=args.experiment,
        alias=args.alias,
        tracking_uri=args.tracking_uri,
        register=not args.no_register,
    )

    print("\n=== ĐĂNG KÝ MODEL ===")
    print(f"run_id     : {result['run_id']}")
    print(f"model_uri  : {result['model_uri']}")
    if result["version"]:
        print(f"version    : {result['version']}  (alias: {result['alias']})")
    print(f"metrics    : {len(result['metrics'])} chỉ số")
    for key in ("best_val_mauroc", "test_mauroc"):
        if key in result["metrics"]:
            print(f"  {key:<16}: {result['metrics'][key]:.4f}")

    _print_versions(args.model_name)
    print(f"\nMLflow UI: {args.tracking_uri or 'http://localhost:5000'}/#/models/{args.model_name}/versions")

    if result["version"] and not args.no_verify:
        if not _verify(args.ckpt, args.model_name, result["alias"]):
            sys.exit(1)


if __name__ == "__main__":
    main()
