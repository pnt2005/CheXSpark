"""DAG: CheXmask → Silver masked (join mask phân đoạn phổi bằng Spark).

Nhiệm vụ: đọc CheXmask RLE (2GB, 112k dòng) từ Bronze, **filter về 2k ảnh dev
ngay lúc đọc** (tránh OOM trên cluster 8GB), broadcast join với Silver, validate
Dice/RLE rồi ghi `s3a://silver/cxr14_masked` (33 cột = 25 nhãn + 7 mask).

Script: `src/ingestion/ingest_chexmask_to_silver.py`
Điều kiện: Silver `cxr14` đã tồn tại (chạy DAG `chexspark_bronze_to_silver` trước).
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
    "execution_timeout": timedelta(minutes=45),
}

with DAG(
    dag_id="chexspark_chexmask_join",
    description="Bronze CheXmask -> Silver cxr14_masked (mask join, Spark)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["chexspark", "silver", "spark", "mask"],
    doc_md=__doc__,
) as dag:
    join_mask = spark_job(
        task_id="ingest_chexmask_to_silver",
        script="ingestion/ingest_chexmask_to_silver.py",
    )

    join_mask
