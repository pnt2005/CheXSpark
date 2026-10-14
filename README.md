# CheXSpark

Xây dựng hệ thống phát hiện và trực quan hóa đa bệnh lý lồng ngực từ ảnh X-quang ngực (CXR) bằng Deep Learning và Apache Spark.

**Pipeline 3 giai đoạn**: Lung Segmentation (U-Net) → Multi-label Classification (DenseNet121) → Explainability (Grad-CAM).

## Kiến trúc hạ tầng (docker-compose)

| Service | Image | Port(s) | Vai trò |
|---|---|---|---|
| **minio** | quay.io/minio/minio | 9000 (S3), 9001 (console) | Object storage: ảnh CXR, mask CheXmask, Silver/Gold datasets |
| **minio-init** | quay.io/minio/mc | — | Tạo bucket `mlflow` tự động khi khởi động |
| **postgres** | postgres:16-alpine | 5432 | Metadata catalog + backend store cho MLflow |
| **spark-master** | apache/spark:3.5.1 | 7077, 8080 | Spark standalone master |
| **spark-worker-1** | apache/spark:3.5.1 | 8081 | Worker (cores/memory cấu hình qua `.env`) |
| **spark-worker-2** | apache/spark:3.5.1 | 8082 | Worker |
| **mlflow** | ghcr.io/mlflow/mlflow | 5000 | Experiment tracking + model registry |

## Cấu trúc repo

```
CheXSpark/
├── docker-compose.yml            # MinIO + PostgreSQL + Spark cluster + MLflow
├── .env.example                  # Biến môi trường (copy → .env trước khi chạy)
├── infra/
│   ├── postgres/init/            # SQL bootstrap (tạo DB mlflow, chexspark_metadata)
│   └── spark/conf/               # spark-defaults.conf (S3/MinIO, MLflow URI)
├── src/
│   ├── ingestion/                # Spark Core: đọc CSV metadata + ảnh từ MinIO → Bronze
│   ├── silver/                   # Spark SQL: cleaning, stratified split → Silver (Parquet)
│   ├── gold/                     # GraphX co-occurrence, MLlib PCA/K-Means/LR/RF → Gold
│   ├── dl/                       # U-Net, DenseNet121, Grad-CAM (PyTorch)
│   └── serving/                  # FastAPI inference + Streamlit UI
├── notebooks/                    # Exploration & visualization notebooks
├── tests/                        # Unit & integration tests
└── scripts/                      # Helper scripts (submit jobs, seed data…)
```

## Cách chạy

```powershell
# 1. Tạo file .env từ template
Copy-Item .env.example .env

# 2. Khởi động toàn bộ hạ tầng
docker compose up -d

# 3. Kiểm tra trạng thái
docker compose ps
```

### Truy cập các UI

- **MinIO Console**: http://localhost:9001 (user/pass từ `.env`)
- **Spark Master UI**: http://localhost:8080
- **Spark Worker UIs**: http://localhost:8081, http://localhost:8082
- **MLflow UI**: http://localhost:5000

### Gửi một Spark job thử

```powershell
docker compose exec spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 `
  --deploy-mode client `
  /opt/spark/user-src/ingestion/example_job.py
```

### Dọn dẹp

```powershell
docker compose down          # dừng services, giữ data volumes
docker compose down -v       # xóa hoàn toàn (kể cả MinIO/Postgres data)
```

## Ghi chú

- Bucket `mlflow` được tạo tự động bởi service `minio-init`; các bucket dữ liệu (`bronze`, `silver`, `gold`, `checx`) tạo qua MinIO console hoặc `mc`.
- Credential MinIO được truyền vào Spark/MLflow qua biến môi trường `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — **không** hard-code trong code hay `spark-defaults.conf`.
- Đảm cấp Docker Desktop có ≥ 8 GB RAM để chạy đủ 2 Spark workers (mỗi worker mặc định 4G, chỉnh trong `.env`).
