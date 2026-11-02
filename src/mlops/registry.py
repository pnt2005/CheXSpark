"""Đăng ký và nạp model Stage-1 (U-Net) / Stage-2 (DenseNet121) qua MLflow.

Vì sao dùng Model Registry thay vì trỏ thẳng file `.pth`:

  - Mỗi lần train sinh ra một **version bất biến**, gắn với đúng run/params/metrics
    của lần train đó → biết chính xác model đang chạy trong serving là bản nào.
  - **Alias** (`production`, `staging`) cho phép promote/rollback mà không sửa code:
    serving luôn đọc `models:/<tên>@production`.
  - Artifact nằm trong MinIO (`s3://mlflow`) nên không phụ thuộc file local hay
    notebook Kaggle.

Hai registered model:

    chexspark-stage1-unet            Stage-1 U-Net phân đoạn phổi
    chexspark-stage2-densenet121     Stage-2 DenseNet121 phân loại 14 bệnh

Luồng dùng::

    # Đăng ký + promote + tự kiểm chứng
    python scripts/dev/register_model.py --stage 1
    python scripts/dev/register_model.py --stage 2

    # Serving nạp model theo alias
    from mlops.registry import load_stage1_from_registry, load_stage2_from_registry
    unet = load_stage1_from_registry()        # models:/chexspark-stage1-unet@production
    densenet = load_stage2_from_registry()    # models:/chexspark-stage2-densenet121@production
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
from dl.segmentation import load_stage1_model

STAGE1_MODEL_NAME = "chexspark-stage1-unet"
STAGE1_EXPERIMENT = "chexspark-stage1-registry"
MODEL_NAME = "chexspark-stage2-densenet121"
EXPERIMENT = "chexspark-stage2-registry"
PRODUCTION_ALIAS = "production"

# Kích thước đầu vào cố định (Stage-2: RGB 3 kênh; Stage-1: xám 1 kênh)
INPUT_SHAPE = (1, 3, IMG_SIZE, IMG_SIZE)
STAGE1_INPUT_SHAPE = (1, 1, IMG_SIZE, IMG_SIZE)

# Model dùng torchvision ở chế độ inference → chỉ cần các gói này, không pin cứng
# bản CPU/CUDA để registry còn dùng được trên máy GPU.
PIP_REQUIREMENTS = ["torch>=2.0", "torchvision>=0.15", "numpy"]


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
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


def _flatten_classification_metrics(raw: dict) -> tuple[dict[str, float], dict[str, str]]:
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


def _flatten_segmentation_metrics(raw: dict) -> tuple[dict[str, float], dict[str, str]]:
    """Tách `stage1_metrics.json` (best_val_dice + test + config) thành metrics/params."""
    metrics: dict[str, float] = {}
    params: dict[str, str] = {}

    if "best_val_dice" in raw:
        metrics["best_val_dice"] = float(raw["best_val_dice"])
    for key, value in (raw.get("test") or {}).items():
        if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
            metrics[f"test_{key}"] = float(value)
    for key, value in (raw.get("config") or {}).items():
        if isinstance(value, (int, float, str, bool)):
            params[f"train_{key}"] = str(value)

    return metrics, params


def _log_and_register(
    ckpt: Path,
    model: torch.nn.Module,
    *,
    metrics: dict[str, float],
    params: dict[str, str],
    tags: dict[str, str],
    example_input: torch.Tensor,
    model_name: str,
    experiment: str,
    alias: str,
    run_name: str,
    model_description: str,
    tracking_uri: str | None,
    register: bool,
) -> dict:
    """Log run + model (kèm signature) rồi gán alias. Dùng chung cho mọi stage."""
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)

        with torch.no_grad():
            example_out = model(example_input)
        signature = infer_signature(example_input.numpy(), example_out.numpy())

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
    client.update_registered_model(model_name, description=model_description)
    client.update_model_version(
        model_name,
        version,
        description=(
            f"Train từ {ckpt.name} (sha256 {params['checkpoint_sha256'][:12]}…)."
        ),
    )
    for key, value in {
        "run_id": result["run_id"],
        "checkpoint_sha256": params["checkpoint_sha256"],
        **{k: f"{v:.4f}" for k, v in metrics.items() if k.startswith(("test_", "best_val_"))},
    }.items():
        client.set_model_version_tag(model_name, version, key, value)

    result.update({"version": version, "alias": alias, "model_uri": f"models:/{model_name}@{alias}"})
    return result


# --------------------------------------------------------------------------- #
# Stage-2: DenseNet121 phân loại 14 bệnh
# --------------------------------------------------------------------------- #
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
    """Log checkpoint Stage-2 vào MLflow và (tuỳ chọn) đăng ký model version mới."""
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt}")

    raw_metrics: dict = {}
    if metrics_json and Path(metrics_json).exists():
        raw_metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    metrics, train_params = _flatten_classification_metrics(raw_metrics)

    params = {
        "arch": "densenet121",
        "n_classes": str(len(DISEASES)),
        "img_size": str(IMG_SIZE),
        "mask_mode": train_params.get("train_mask_mode", "lungs"),
        "dataset": "NIH ChestX-ray14 (dev subset 2000 ảnh)",
        "n_train": "1335",
        "n_val": "321",
        "n_test": "344",
        "checkpoint": ckpt.name,
        "checkpoint_sha256": _sha256(ckpt),
        **train_params,
    }

    torch.manual_seed(0)
    return _log_and_register(
        ckpt,
        load_stage2_model(ckpt),
        metrics=metrics,
        params=params,
        tags={
            "stage": "stage2-classification",
            "task": "multi-label 14 bệnh lồng ngực",
            "framework": "pytorch",
            "input": "ảnh CXR 224×224 RGB × mask phổi",
            "output": "14 xác suất (sigmoid)",
        },
        example_input=torch.randn(*INPUT_SHAPE),
        model_name=model_name,
        experiment=experiment,
        alias=alias,
        run_name=f"stage2-densenet121-{params['checkpoint_sha256'][:8]}",
        model_description=(
            "DenseNet121 đa nhãn 14 bệnh lồng ngực (giai đoạn 2 của CheXSpark). "
            "Đầu vào: ảnh X-quang 224×224×3 đã áp mask phổi (U-Net Stage-1 hoặc CheXmask). "
            "Đầu ra: 14 logits (áp sigmoid để ra xác suất)."
        ),
        tracking_uri=tracking_uri,
        register=register,
    )


# --------------------------------------------------------------------------- #
# Stage-1: U-Net phân đoạn phổi
# --------------------------------------------------------------------------- #
def register_stage1_model(
    ckpt: str | Path,
    metrics_json: str | Path | None = None,
    *,
    model_name: str = STAGE1_MODEL_NAME,
    experiment: str = STAGE1_EXPERIMENT,
    alias: str = PRODUCTION_ALIAS,
    tracking_uri: str | None = None,
    register: bool = True,
) -> dict:
    """Log checkpoint U-Net Stage-1 vào MLflow và (tuỳ chọn) đăng ký model version mới.

    Args:
        ckpt: file `.pth` (state_dict U-Net, 1 kênh vào / 1 kênh ra)
        metrics_json: `stage1_metrics.json` sinh bởi trainer/notebook Kaggle
        register: False → chỉ log run, không tạo model version
    """
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Không thấy checkpoint: {ckpt}")

    raw_metrics: dict = {}
    if metrics_json and Path(metrics_json).exists():
        raw_metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    metrics, train_params = _flatten_segmentation_metrics(raw_metrics)

    params = {
        "arch": "unet",
        "in_channels": "1",
        "n_classes": "1 (phổi)",
        "img_size": str(IMG_SIZE),
        "dataset": "NIH ChestX-ray14 (dev subset 2000 ảnh) + CheXmask ground truth",
        "ground_truth": "CheXmask (left|right, không gồm heart)",
        "checkpoint": ckpt.name,
        "checkpoint_sha256": _sha256(ckpt),
        **train_params,
    }

    torch.manual_seed(0)
    return _log_and_register(
        ckpt,
        load_stage1_model(ckpt),
        metrics=metrics,
        params=params,
        tags={
            "stage": "stage1-segmentation",
            "task": "phân đoạn trường phổi (binary)",
            "framework": "pytorch",
            "input": "ảnh CXR xám 224×224×1 (chuẩn hoá [-1,1])",
            "output": "logits mask phổi 224×224 (sigmoid + ngưỡng 0.5)",
        },
        example_input=torch.randn(*STAGE1_INPUT_SHAPE),
        model_name=model_name,
        experiment=experiment,
        alias=alias,
        run_name=f"stage1-unet-{params['checkpoint_sha256'][:8]}",
        model_description=(
            "U-Net phân đoạn trường phổi (giai đoạn 1 của CheXSpark). "
            "Đầu vào: ảnh X-quang xám 224×224×1. "
            "Đầu ra: logits mask phổi 224×224 (sigmoid, ngưỡng 0.5). "
            "Thay CheXmask ở thời điểm suy luận."
        ),
        tracking_uri=tracking_uri,
        register=register,
    )


# --------------------------------------------------------------------------- #
# Promote / nạp model / liệt kê
# --------------------------------------------------------------------------- #
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
    """Nạp DenseNet121 đang gắn alias (mặc định `production`) từ registry."""
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model = mlflow.pytorch.load_model(f"models:/{model_name}@{alias}", map_location=device)
    model.eval()
    return model.to(device)


def load_stage1_from_registry(
    model_name: str = STAGE1_MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
    *,
    tracking_uri: str | None = None,
    device: str = "cpu",
):
    """Nạp U-Net phân đoạn đang gắn alias (mặc định `production`) từ registry."""
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
    """Liệt kê version + alias + toàn bộ metrics của một registered model."""
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
                "metrics": run_metrics,
                "created": mv.creation_timestamp,
            }
        )
    return rows
