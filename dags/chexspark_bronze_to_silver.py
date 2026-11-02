"""DAG: Bronze → Silver (metadata + nhãn 14 bệnh) bằng Spark Core.

Nhiệm vụ: đọc 112,120 dòng metadata NIH từ Bronze (MinIO), join manifest 2k ảnh
dev, tạo multi-hot 14 nhãn, chia Train/Val/Test theo **patient-level** (chống rò
rỉ bệnh nhân), ghi Silver Parquet partition theo split và log thống kê lên MLflow.

Script: `src/ingestion/ingest_bronze_to_silver.py`
Output: `s3a://silver/cxr14/split={train,val,test}/`
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG

from spark_utils import spark_job

default_args = {
    "owner": "chexspark",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="chexspark_bronze_to_silver",
    description="Bronze metadata 112k -> Silver cxr14 (Spark Core + MLflow)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["chexspark", "silver", "spark"],
    doc_md=__doc__,
) as dag:
    ingest = spark_job(
        task_id="ingest_bronze_to_silver",
        script="ingestion/ingest_bronze_to_silver.py",
    )

    ingest
