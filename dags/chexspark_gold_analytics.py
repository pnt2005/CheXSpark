"""DAG: Gold analytics — co-occurrence, mạng bệnh lý, verify.

Chuỗi 3 job Spark (chạy tuần tự vì phụ thuộc dữ liệu):
  1. `disease_cooccurrence.py` — ma trận đồng xuất hiện 14×14 trên **112,120 ảnh**,
     Jaccard + xác suất có điều kiện, comorbidity, thống kê dịch tễ
  2. `disease_graph.py` — mạng bệnh lý: lift = P(d1,d2)/(P(d1)P(d2)), lọc cạnh
     theo ngưỡng lift chọn bằng modularity, greedy modularity (CNM) → cộng đồng bệnh
  3. `verify_gold.py` — đọc lại toàn bộ output Gold, in tóm tắt để kiểm chứng

Output: `s3a://gold/{cooccurrence,stats,graph}/`
MLflow: experiment `chexspark-gold-cooccurrence`, `chexspark-gold-graph`
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
    "execution_timeout": timedelta(minutes=40),
}

with DAG(
    dag_id="chexspark_gold_analytics",
    description="Gold: co-occurrence 14x14 + disease network + verify (Spark)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["chexspark", "gold", "spark", "graph"],
    doc_md=__doc__,
) as dag:
    cooccurrence = spark_job(
        task_id="disease_cooccurrence",
        script="gold/disease_cooccurrence.py",
    )
    graph = spark_job(
        task_id="disease_graph",
        script="gold/disease_graph.py",
    )
    verify = spark_job(
        task_id="verify_gold",
        script="gold/verify_gold.py",
    )

    cooccurrence >> graph >> verify
