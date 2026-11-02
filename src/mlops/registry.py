"""Đăng ký và nạp model Stage-2 qua MLflow Model Registry.

Vì sao dùng Model Registry thay vì trỏ thẳng file `.pth`:

  - Mỗi lần train sinh ra một **version bất biến**, gắn với đúng run/params/metrics
    của lần train đó → biết chính xác model đang chạy trong serving là bản nào.
  - **Alias** (`production`, `staging`) cho phép promote/rollback mà không sửa code:
    serving luôn đọc `models:/chexspark-stage2-densenet121@production`.
  - Artifact nằm trong MinIO (`s3://mlflow`) nên không phụ thuộc file local hay
    notebook Kaggle.

Luồng dùng::

    # 1) Đăng ký checkpoint + promote lên alias production
    python scripts/dev/register_stage2_model.py

    # 2) Serving nạp model từ registry
    from mlops.registry import load_stage2_from_registry
    model = load_stage2_from_registry()          # -> models:/...@production
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from dl.model import load_stage2_model
from dl.preprocess import DISEASES, IMG_SIZE

MODEL_NAME = "chexspark-stage2-densenet121"
EXPERIMENT = "chexspark-stage2-registry"
PRODUCTION_ALIAS = "production"

# Kiến trúc/kích thước đầu vào cố định của Stage-2 (DenseNet121, ảnh 224×224 RGB)
INPUT_SHAPE = (1, 3, IMG_SIZE, IMG_SIZE)

# Model dùng torchvision ở chế độ inference → chỉ cần 2 gói này, không pin cứng
# bản CPU/CUDA để registry còn dùng được trên máy GPU.
PIP_REQUIREMENTS = ["torch>=2.0", "torchvision>=0.15", "numpy"]


def _sha256(path: str | Path) -> str:
    """Hash checkpoint — để truy vết version registry ↔ file .pth gốc."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version_for_run(client, model_name: str, run_id: str) -> str:
    """Tìm model version vừa được tạo từ một run (fallback cho MLflow cũ)."""
    versions = [mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id]
    if not versions:
        raise RuntimeError(f"Không tìm thấy version nào của {model_name} gắn với run {run_id}")
    return max(versions, key=lambda mv: int(mv.version)).version


def _flatten_metrics(raw: dict) -> tuple[dict[str, float], dict[str, str]]:
    """Tách `stage2_metrics.json` thành metrics (số) + params (cấu hình train)."""
    metrics: dict[str, float] = {}
    params: dict[str, str] = {}

    if "best_val_auc" in raw:
        metrics["best_val_mauroc"] = float(raw["best_val_auc"])
    if "test_auc" in raw:
        metrics["test_mauroc"] = float(raw["test_auc"])

    # AUC từng bệnh: Hernia = null (không có ca dương tính trong test) → bỏ qua
    for disease, auc in (raw.get("per_label_auc") or {}).items():
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            continue
        metrics[f"auc_{disease}"] = float(auc)

    for key, value in (raw.get("config") or {}).items():
        params[f"train_{key}"] = str(value)

    return metrics, params


def register_stage2_model(
    ckpt: str | Path,
    metrics_json: str | Path | None = None,
    *,
    model_name: str = MODEL_NAME,
    experiment: str = EXPERIMENT,
    alias: str = PRODUCTION_ALIAS,
    tracking_uri: str | None = None,
    register: bool = True,
) -> dict:
    """Log checkpoint vào MLflow và (tuỳ chọn) đăng ký thành model version mới.

    Args:
        ckpt: file `.pth` (state_dict DenseNet121, classifier 14 lớp)
        metrics_json: `stage2_metrics.json` sinh bởi notebook Kaggle
        alias: alias gán cho version mới (mặc định `production`)
        register: False → chỉ log run, không tạo model version

    Returns:
        dict gồm run_id, version, alias, model_uri, metrics.
    """
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt}")

    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment)

    model = load_stage2_model(ckpt)

    raw_metrics: dict = {}
    if metrics_json and Path(metrics_json).exists():
        raw_metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    metrics, train_params = _flatten_metrics(raw_metrics)

    params = {
        "arch": "densenet121",
        "n_classes": str(len(DISEASES)),
        "img_size": str(IMG_SIZE),
        "mask_mode": train_params.get("train_mask_mode", "lungs"),
        "seed": "42",
        "dataset": "NIH ChestX-ray14 (dev subset 2000 ảnh)",
        "n_train": "1335",
        "n_val": "321",
        "n_test": "344",
        "checkpoint": ckpt.name,
        "checkpoint_sha256": _sha256(ckpt),
        **train_params,
    }

    with mlflow.start_run(run_name=f"stage2-densenet121-{params['checkpoint_sha256'][:8]}") as run:
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "stage": "stage2-classification",
                "task": "multi-label 14 bệnh lồng ngực",
                "framework": "pytorch",
                "input": "ảnh CXR 224×224 RGB × mask phổi CheXmask",
                "output": "14 xác suất (sigmoid)",
            }
        )

        # Signature: input (1,3,224,224) float32 → output (1,14) logits.
        torch.manual_seed(0)
        example = torch.randn(*INPUT_SHAPE)
        with torch.no_grad():
            example_out = model(example)
        signature = infer_signature(example.numpy(), example_out.numpy())

        info = mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name if register else None,
            pip_requirements=PIP_REQUIREMENTS,
        )

        result = {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "version": None,
            "alias": None,
            "model_uri": info.model_uri,
            "metrics": metrics,
        }

    if not register:
        return result

    client = MlflowClient()
    # Một số bản MLflow không điền `registered_model_version` → tra theo run_id
    version = str(
        getattr(info, "registered_model_version", None)
        or _version_for_run(client, model_name, result["run_id"])
    )

    # Alias là cách promote hiện đại của MLflow (stages đã deprecated từ 2.9).
    client.set_registered_model_alias(model_name, alias, version)
    client.update_registered_model(
        model_name,
        description=(
            "DenseNet121 đa nhãn 14 bệnh lồng ngực (giai đoạn 2 của CheXSpark). "
            "Đầu vào: ảnh X-quang 224×224×3 đã áp mask phổi CheXmask. "
            "Đầu ra: 14 logits (áp sigmoid để ra xác suất)."
        ),
    )
    client.update_model_version(
        model_name,
        version,
        description=(
            f"Train từ {ckpt.name} (sha256 {params['checkpoint_sha256'][:12]}…); "
            f"val mAUROC {metrics.get('best_val_mauroc', float('nan')):.4f}, "
            f"test mAUROC {metrics.get('test_mauroc', float('nan')):.4f}."
        ),
    )
    for key, value in {
        "run_id": run.info.run_id,
        "checkpoint_sha256": params["checkpoint_sha256"],
        "test_mauroc": f"{metrics.get('test_mauroc', float('nan')):.4f}",
    }.items():
        client.set_model_version_tag(model_name, version, key, value)

    result.update(
        {
            "version": version,
            "alias": alias,
            "model_uri": f"models:/{model_name}@{alias}",
        }
    )
    return result


def promote(model_name: str, version: str | int, alias: str = PRODUCTION_ALIAS) -> str:
    """Trỏ alias sang một version khác (promote/rollback không cần sửa code)."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, str(version))
    return f"models:/{model_name}@{alias}"


def load_stage2_from_registry(
    model_name: str = MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
    *,
    tracking_uri: str | None = None,
    device: str = "cpu",
):
    """Nạp model đang được gắn alias (mặc định `production`) từ registry."""
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model = mlflow.pytorch.load_model(f"models:/{model_name}@{alias}", map_location=device)
    model.eval()
    return model.to(device)


def _aliases_by_version(model_name: str) -> dict[str, list[str]]:
    """Map version → [alias] qua REST API.

    Dataclass `ModelVersion` của MLflow 2.9 không có field `aliases` (aliases chỉ
    có trong response REST), nên phải đọc trực tiếp endpoint registered-models/get.
    """
    import requests

    uri = mlflow.get_tracking_uri().rstrip("/")
    try:
        resp = requests.get(
            f"{uri}/api/2.0/mlflow/registered-models/get",
            params={"name": model_name},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception:
        return {}

    mapping: dict[str, list[str]] = {}
    for item in (resp.json().get("registered_model") or {}).get("aliases") or []:
        mapping.setdefault(str(item["version"]), []).append(item["alias"])
    return mapping


def describe(model_name: str = MODEL_NAME) -> list[dict]:
    """Liệt kê version + alias + metrics của một registered model."""
    client = MlflowClient()
    aliases = _aliases_by_version(model_name)

    rows = []
    for mv in sorted(client.search_model_versions(f"name='{model_name}'"), key=lambda m: int(m.version)):
        run_metrics = {}
        if mv.run_id:
            run_metrics = client.get_run(mv.run_id).data.metrics
        rows.append(
            {
                "version": mv.version,
                "aliases": aliases.get(str(mv.version), []),
                "run_id": mv.run_id,
                "test_mauroc": run_metrics.get("test_mauroc"),
                "best_val_mauroc": run_metrics.get("best_val_mauroc"),
                "created": mv.creation_timestamp,
            }
        )
    return rows
