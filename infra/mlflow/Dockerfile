# CheXSpark MLflow image — bổ sung driver cho backend store + artifact store
#
# Image chính thức `ghcr.io/mlflow/mlflow` chỉ có mlflow + sqlalchemy, THIẾU:
#   - psycopg2 → không dùng được `--backend-store-uri postgresql://…`
#   - boto3    → không ghi được artifact lên MinIO (`s3://mlflow`)
# Thiếu 2 gói này thì MLflow âm thầm rơi về sqlite + filesystem trong container:
# metadata và artifact mất sạch mỗi lần `docker compose up --force-recreate`.
FROM ghcr.io/mlflow/mlflow:v2.9.2

RUN pip install --no-cache-dir \
        "psycopg2-binary==2.9.9" \
        "boto3==1.34.34"
