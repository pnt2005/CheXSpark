"""DAG: CheXSpark full Medallion pipeline (orchestration).

Chạy toàn bộ chuỗi bằng cách trigger các DAG con và **chờ** từng cái xong:

    Bronze → Silver (metadata)  →  Silver masked (CheXmask join)  →  Gold analytics

Mỗi bước là một DAG riêng (tách để có thể chạy lại độc lập khi lỗi), DAG này chỉ
điều phối thứ tự. Trigger thủ công từ UI/CLI:

    airflow dags trigger chexspark_pipeline
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "chexspark",
    "depends_on_past": False,
    "retries": 0,
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="chexspark_pipeline",
    description="Medallion pipeline: Bronze -> Silver -> Silver masked -> Gold",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["chexspark", "pipeline", "medallion"],
    doc_md=__doc__,
) as dag:
    bronze_to_silver = TriggerDagRunOperator(
        task_id="run_bronze_to_silver",
        trigger_dag_id="chexspark_bronze_to_silver",
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=True,
    )

    chexmask_join = TriggerDagRunOperator(
        task_id="run_chexmask_join",
        trigger_dag_id="chexspark_chexmask_join",
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=True,
    )

    gold_analytics = TriggerDagRunOperator(
        task_id="run_gold_analytics",
        trigger_dag_id="chexspark_gold_analytics",
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=True,
    )

    bronze_to_silver >> chexmask_join >> gold_analytics
