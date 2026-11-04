# CheXSpark Serving image — FastAPI (REST inference) + Streamlit (dashboard)
#
# Một image dùng cho CẢ hai service (khác nhau chỉ ở command): cùng model, cùng
# code, không phải build 2 lần.
#
# torch CPU (không CUDA): suy luận 224×224 ~0.3s/ảnh trên CPU là đủ cho demo;
# máy dev không có GPU, còn train vẫn chạy trên Kaggle GPU như trước.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    CHEXSPARK_ROOT=/app \
    HF_HUB_OFFLINE=1

WORKDIR /app

# 1) dependency của torch trước từ PyPI
# 2) torch/torchvision bản CPU từ index pytorch — chỉ 200MB thay vì ~2.5GB (CUDA)
#
# PIN torch==2.5.1: từ torch 2.6 `torch.load` đổi mặc định weights_only=True →
# `mlflow.pytorch.load_model` (nạp model pickled đầy đủ) thất bại với
# "Weights only load failed". MLflow pin 2.9.2 nên không tự nâng torch được.
RUN pip install --no-cache-dir \
        "numpy==1.26.4" "filelock" "typing-extensions" "sympy" \
        "networkx" "jinja2" "fsspec" && \
    pip install --no-cache-dir "torch==2.5.1" "torchvision==0.20.1" \
        --index-url https://download.pytorch.org/whl/cpu

# FastAPI (API) + Streamlit (UI) + mlflow/boto3/pyarrow (nạp model registry,
# đọc Gold layer từ MinIO) + scipy (ước lượng vùng tim cho mask_input=3ch)
RUN pip install --no-cache-dir \
        "fastapi==0.115.6" "uvicorn[standard]==0.34.0" "python-multipart==0.0.20" \
        "streamlit==1.41.1" "plotly==5.24.1" \
        "mlflow==2.9.2" "boto3==1.34.131" "pyarrow==14.0.2" "pandas==2.2.3" \
        "pillow==10.4.0" "scipy==1.13.1" "requests==2.32.3"

# Code: image tự chứa bản sao để chạy standalone; docker-compose mount đè
# `./src:/app/src:ro` nên chỉnh code không cần rebuild.
COPY src /app/src

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000 8501
# Mặc định chạy FastAPI; service streamlit-app override command
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
