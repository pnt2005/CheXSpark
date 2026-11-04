"""FastAPI — REST API phục vụ chẩn đoán CXR (3 giai đoạn) + analytics Gold layer.

Endpoints
---------
``GET  /api/v1/health``                      trạng thái service + Gold layer
``GET  /api/v1/model``                       model đang serve (alias, calibration, version registry)
``POST /api/v1/predict``                     ảnh CXR → 14 xác suất + mask + heatmap (base64 PNG)
``GET  /api/v1/analytics/disease-distribution``  phân bố bệnh từ Gold (Spark/GraphX)
``GET  /api/v1/analytics/top-pairs``         top cặp đồng xuất hiện
``GET  /api/v1/graph/co-occurrence``         nodes + edges + summary mạng bệnh lý

Chạy (host, đã cài fastapi/uvicorn/python-multipart)::

    uvicorn serving.api:app --host 0.0.0.0 --port 8000      # PYTHONPATH=src

hoặc qua Docker Compose (service `fastapi-app`, port 8000).
"""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from dl.preprocess import DISEASES
from serving.gold_store import GoldUnavailable, gold_status, read_gold
from serving.pipeline import InferenceEngine

MAX_UPLOAD_MB = float(os.getenv("SERVING_MAX_UPLOAD_MB", "25"))
API_PREFIX = "/api/v1"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class Finding(BaseModel):
    disease_name: str
    probability: float = Field(..., description="Xác suất đã qua temperature scaling")
    threshold: float = Field(..., description="Ngưỡng Youden fit trên val")
    is_positive: bool


class HealthResponse(BaseModel):
    status: str
    device: Optional[str] = None
    model_loaded: bool
    gold: dict


class PredictResponse(BaseModel):
    findings: list[Finding]
    top_disease: str
    gradcam_disease: str
    mask_coverage: float
    images: dict[str, Optional[str]] = Field(
        ..., description="PNG base64: original / mask / masked / heatmap (224×224)"
    )
    timings_ms: dict[str, float]
    model: dict


# --------------------------------------------------------------------------- #
# App lifecycle — nạp model 1 lần khi khởi động
# --------------------------------------------------------------------------- #
_engine: InferenceEngine | None = None
_engine_error: str | None = None


def _init_engine() -> None:
    global _engine, _engine_error
    try:
        _engine = InferenceEngine()
        _engine_error = None
        print(f"[serving] engine sẵn sàng trên {_engine.device} — {_engine.stage2_info}")
    except Exception as exc:  # model không nạp được → /predict trả 503 rõ ràng
        _engine = None
        _engine_error = f"{type(exc).__name__}: {exc}"
        print(f"[serving] KHÔNG nạp được model: {_engine_error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(_init_engine)
    yield


app = FastAPI(
    title="CheXSpark Serving API",
    version="1.0.0",
    description=(
        "Phục vụ pipeline 3 giai đoạn CheXSpark: U-Net phân đoạn phổi → "
        "DenseNet121 phân loại 14 bệnh lồng ngực → Grad-CAM giải thích. "
        "Kèm endpoint analytics đọc từ Gold layer (Spark/GraphX trên MinIO)."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_engine() -> InferenceEngine:
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model chưa sẵn sàng: {_engine_error or 'đang khởi tạo'}",
        )
    return _engine


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _png_b64(arr: np.ndarray | None) -> str | None:
    """Mảng uint8/float [0,1] → base64 PNG (224×224)."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 2:
        img = Image.fromarray(a, mode="L")
    else:
        img = Image.fromarray(a)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gold_rows(prefix: str) -> list[dict]:
    try:
        return read_gold(prefix)
    except GoldUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Health / model
# --------------------------------------------------------------------------- #
@app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """503 cho tới khi model nạp xong → healthcheck của container báo unhealthy."""
    if _engine is None:
        response.status_code = 503
    return HealthResponse(
        status="ok" if _engine is not None else "degraded",
        device=_engine.device if _engine else None,
        model_loaded=_engine is not None,
        gold=gold_status(),
    )


@app.get(f"{API_PREFIX}/model")
def model_info() -> dict:
    """Thông tin model đang serve + (best-effort) toàn bộ version trong registry."""
    engine = _require_engine()
    info = dict(engine.info)
    try:
        from mlops.registry import MODEL_NAME, STAGE1_MODEL_NAME, describe

        info["registry"] = {
            MODEL_NAME: describe(MODEL_NAME),
            STAGE1_MODEL_NAME: describe(STAGE1_MODEL_NAME),
        }
    except Exception as exc:  # MLflow chưa chạy → vẫn trả info local
        info["registry"] = {"error": f"{type(exc).__name__}: {exc}"}
    return info


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
@app.post(f"{API_PREFIX}/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Ảnh CXR (PNG/JPEG/DICOM-đã-chuyển PNG)"),
    disease: Optional[str] = Query(
        None, description="Bệnh cần vẽ Grad-CAM (mặc định: bệnh có xác suất cao nhất)"
    ),
    gradcam: bool = Query(True, description="Tắt để chỉ lấy xác suất (nhanh hơn ~300ms)"),
) -> PredictResponse:
    engine = _require_engine()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File rỗng")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File {len(data) / 1e6:.1f}MB vượt giới hạn {MAX_UPLOAD_MB}MB",
        )

    try:
        result = await run_in_threadpool(
            engine.predict, data, disease=disease, want_gradcam=gradcam
        )
    except ValueError as exc:  # tên bệnh sai, ảnh không mở được…
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận: {exc}") from exc

    result["images"] = {k: _png_b64(v) for k, v in result["images"].items()}
    return PredictResponse(**result)


# --------------------------------------------------------------------------- #
# Analytics (Gold layer — do Spark/GraphX ghi)
# --------------------------------------------------------------------------- #
@app.get(f"{API_PREFIX}/analytics/disease-distribution")
def disease_distribution() -> dict:
    """Phân bố 14 bệnh + phân bố số bệnh/ảnh trên toàn bộ 112k ảnh NIH."""
    return {
        "diseases": _gold_rows("stats/disease"),
        "comorbidity": _gold_rows("stats/comorbidity"),
        "source": "s3a://gold/stats/{disease,comorbidity}",
    }


@app.get(f"{API_PREFIX}/analytics/top-pairs")
def top_pairs(limit: int = Query(20, ge=1, le=100)) -> dict:
    rows = _gold_rows("stats/top_pairs")
    return {"pairs": rows[:limit], "source": "s3a://gold/stats/top_pairs"}


@app.get(f"{API_PREFIX}/graph/co-occurrence")
def cooccurrence_graph() -> dict:
    """Mạng đồng xuất hiện bệnh lý: nodes (degree/centrality/community) + edges + summary."""
    return {
        "nodes": _gold_rows("graph/nodes"),
        "edges": _gold_rows("graph/edges"),
        "summary": _gold_rows("graph/summary"),
        "source": "s3a://gold/graph/{nodes,edges,summary}",
    }


@app.get(f"{API_PREFIX}/diseases")
def diseases() -> dict:
    """Danh sách 14 bệnh (thứ tự khớp output model)."""
    return {"diseases": DISEASES}


@app.exception_handler(GoldUnavailable)
async def gold_unavailable_handler(request, exc: GoldUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})
