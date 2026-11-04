"""DAG: Retrain Stage-2 trên toàn bộ 112k → promote model `production`.

Vòng lặp MLOps, **trigger thủ công** (`schedule=None`) vì mỗi lần tốn ~4 GPU-giờ:

    1. ingest Bronze → Silver full 112k            (Spark)
    2. precompute mask 224 từ Bronze CheXmask      (Spark)
    3. stage Parquet từ MinIO về local             (boto3)
    4. export dataset Kaggle (labels CSV + mask pack .npy)
    5. push train 30 epoch lên Kaggle              (--no-wait)
    6. chờ xong → tải output → đăng ký MLflow      (--wait-only --register)
    7. xác nhận alias `production`

Không đụng vào bản dev 2k: job ghi `silver/cxr14_full` + `silver/cxr14_masks_full`,
dataset Kaggle `chexspark-stage2-nih-112k` / `chexspark-masks-224-112k` — đúng các
đường dẫn đã dùng ở README mục 5c.

Params:
    push_training (bool, mặc định True) — False để chạy thử bước 1–4 (chuẩn bị
    dữ liệu) cho nhanh, không tốn GPU/quota Kaggle; task 5–7 in `[skip]` rồi thành
    công.

Khôi phục: nếu task 6 bị gián đoạn (máy restart / container rơi) thì chạy lại
**task đó** — `--wait-only` không push lại, chỉ poll kernel đang chạy rồi tải +
đăng ký (cơ chế đã dùng để phục hồi run 112k lần đầu).

Chạy tay tương đương: README mục 5c (bước 1–4) + mục Train trên Kaggle (lệnh 6).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from spark_utils import APP_DIR, spark_job

# Nơi docker-compose mount (xem x-airflow-volumes) — cùng layout với host:
#   /opt/airflow/data   ↔ ./data      (dataset Kaggle + output train)
#   /opt/airflow/scripts ↔ ./scripts  (wrapper Kaggle + script export)
DATA_DIR = Path(os.getenv("CHEXSPARK_DATA_DIR", "/opt/airflow/data"))
SCRIPTS_DIR = Path(os.getenv("CHEXSPARK_SCRIPTS_DIR", "/opt/airflow/scripts"))
STAGE_DIR = DATA_DIR / "tmp"
# staging do CHÍNH task này tạo. Không dùng `data/tmp/silver_full` (đã tồn tại từ
# bước chạy tay): thư mục đó 755 thuộc uid khác (do container `mc` tạo) nên user
# airflow không xóa/tạo được file bên trong → PermissionError khi rmtree.
STAGE_ROOT = STAGE_DIR / "retrain_stage"
LABELS_DIR = DATA_DIR / "kaggle/stage2-nih-112k"
MASKS_DIR = DATA_DIR / "kaggle/stage2-masks-224-112k"
OUT_DIR = DATA_DIR / "kaggle/output/stage2-112k"

# Silver full do `ingest_bronze_to_silver.py --full` / `precompute_masks.py --source bronze` ghi
SILVER_FULL = "cxr14_full"
SILVER_MASKS_FULL = "cxr14_masks_full"
S3_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://minio:9000")
S3_BUCKET = "silver"

LABELS_SLUG = "chexspark-stage2-nih-112k"
MASKS_SLUG = "pnt2005/chexspark-masks-224-112k"
EPOCHS = "30"

default_args = {
    "owner": "chexspark",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run(cmd: list) -> None:
    """Chạy subprocess, in lệnh rồi raise nếu thoát mã khác 0 (task fail)."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def _want_training(context) -> bool:
    """False khi param `push_training=False` → task in [skip] rồi báo thành công."""
    if context["params"].get("push_training", True):
        return True
    print("[skip] push_training=False → bỏ qua bước Kaggle/MLflow", flush=True)
    return False


# ─────────────────────────────── callable của task ─────────────────────────── #
def stage_parquet_from_minio() -> None:
    """Sao chép Parquet từ MinIO về local để script export chạy bằng pyarrow/pandas.

    Docker Desktop không cho Spark ghi thẳng ra thư mục host qua executor (executor
    chạy trong container worker riêng), nên lấy về bằng boto3 — credentials đã có
    trong env `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
    """
    import boto3

    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)  # do task này tạo → thuộc uid airflow
    for prefix in (SILVER_FULL, SILVER_MASKS_FULL):
        local = STAGE_ROOT / prefix
        if local.exists():
            try:
                shutil.rmtree(local)  # tránh lẫn part-file cũ của lần chạy trước
            except PermissionError as exc:
                raise PermissionError(
                    f"Không xóa được {local} (thư mục thuộc uid khác, 755). "
                    f"Chạy 1 lần trên host: "
                    f'docker run --rm -v "D:\\BigData\\CheXSpark\\data\\tmp:/d" '
                    f"alpine chmod -R 777 /d") from exc
        local.mkdir(parents=True, exist_ok=True)

        n = 0
        total = 0
        for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=S3_BUCKET, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or not key.endswith(".parquet"):
                    continue
                dest = local / key[len(prefix) + 1:]
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(S3_BUCKET, key, str(dest))
                n += 1
                total += obj["Size"]
        if not n:
            raise FileNotFoundError(
                f"Không thấy Parquet nào trong s3://{S3_BUCKET}/{prefix} — "
                f"chạy task ingest/precompute trước")
        print(f"[stage] s3://{S3_BUCKET}/{prefix} → {local} "
              f"({n} file, {total / 1024 ** 2:.0f} MB)", flush=True)


def export_kaggle_datasets() -> None:
    """Export 2 dataset Kaggle: mask pack .npy (1,34 GB) + labels CSV không RLE."""
    _run([
        sys.executable, SCRIPTS_DIR / "dev/export_masks_to_kaggle.py",
        "--silver-dir", STAGE_ROOT / SILVER_MASKS_FULL,
        "--out-dir", MASKS_DIR,
        "--dataset-id", MASKS_SLUG,
        "--title", "CheXSpark Masks 224 112k (packed)",
    ])
    _run([
        sys.executable, SCRIPTS_DIR / "dev/export_silver_labels.py",
        "--silver-dir", STAGE_ROOT / SILVER_FULL,
        "--out", LABELS_DIR / "stage2_train.csv",
        "--drop", "image_uri", "silver_ingested_at",
    ])


def push_training(**context) -> None:
    """Upload 2 dataset rồi push kernel train — `--no-wait` để thoát ngay sau khi
    push (khâu chờ do task `wait_and_register` đảm nhiệm, tách ra để chạy lại được)."""
    if not _want_training(context):
        return
    _run([
        sys.executable, SCRIPTS_DIR / "kaggle/train_on_kaggle.py",
        "--stage", "2",
        "--dataset-slug", LABELS_SLUG, "--dataset-dir", LABELS_DIR,
        "--upload-dataset",
        "--masks-dataset", MASKS_SLUG, "--masks-dir", MASKS_DIR,
        "--upload-masks",
        "--epochs", EPOCHS,
        "--run-timeout", "43200",
        "--no-wait",
        "--output-dir", OUT_DIR,
    ])


def wait_and_register(**context) -> None:
    """Poll kernel tới khi xong → tải checkpoint/metrics → đăng ký alias `production`.

    Idempotent khi chạy lại: `--wait-only` không push nên không khởi động lần train
    thứ hai (khác với việc chạy lại task `push_training`).
    """
    if not _want_training(context):
        return
    _run([
        sys.executable, SCRIPTS_DIR / "kaggle/train_on_kaggle.py",
        "--stage", "2",
        "--dataset-slug", LABELS_SLUG, "--dataset-dir", LABELS_DIR,
        "--masks-dataset", MASKS_SLUG, "--masks-dir", MASKS_DIR,
        "--epochs", EPOCHS,
        "--wait-only", "--register", "--install",
        "--run-timeout", "43200", "--wait-timeout", "46800",
        "--output-dir", OUT_DIR,
    ])


def verify_production(**context) -> None:
    """Đối chiếu: alias `production` đã trỏ version mới và metrics khớp file tải về."""
    if not _want_training(context):
        return
    sys.path.insert(0, APP_DIR)  # import lười để không tốn thời gian parse DAG
    from mlflow.tracking import MlflowClient

    from mlops.registry import MODEL_NAME

    metrics_path = OUT_DIR / "stage2_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Chưa có {metrics_path} — task trước chưa tải output?")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_mauroc = (metrics.get("test") or {}).get("mauroc")

    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "production")
    print(f"[verify] {MODEL_NAME}@production → version {mv.version} "
          f"(run {mv.run_id}, status {mv.status})", flush=True)
    print(f"[verify] test mAUROC local = {test_mauroc} | best val = "
          f"{metrics.get('best_val_auc')} | epoch {metrics.get('last_epoch')}", flush=True)
    if mv.status != "READY":
        raise RuntimeError(f"Model version {mv.version} chưa READY (status={mv.status})")
    if not isinstance(test_mauroc, (int, float)) or test_mauroc <= 0:
        raise RuntimeError("test mAUROC trong stage2_metrics.json không hợp lệ")


# ──────────────────────────────────── DAG ──────────────────────────────────── #
with DAG(
    dag_id="chexspark_retrain_112k",
    description="Retrain Stage-2 112k: Spark prep -> export -> Kaggle train -> MLflow production",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    params={
        "push_training": Param(
            default=True, type="boolean",
            description="False = chỉ chạy chuẩn bị dữ liệu (task 1-4), không push train",
        ),
    },
    tags=["chexspark", "retrain", "stage2", "kaggle", "mlops"],
    doc_md=__doc__,
) as dag:
    # 1-2. Spark: dựng Silver full + mask pack 112k (việc "nặng" nhưng không tốn GPU)
    ingest_full = spark_job(
        task_id="ingest_silver_full",
        script="ingestion/ingest_bronze_to_silver.py",
        application_args=["--full"],
        execution_timeout=timedelta(minutes=45),
    )

    precompute_masks = spark_job(
        task_id="precompute_masks_full",
        script="silver/precompute_masks.py",
        application_args=[
            "--source", "bronze",
            "--out", f"s3a://silver/{SILVER_MASKS_FULL}",
        ],
        # CheXmask 2GB nằm trong MỘT file → ép chia nhỏ lúc đọc (32MB/partition)
        # thay vì 1 task đọc tuần tự toàn bộ (đo được 457→540 mask/s khi tách).
        extra_conf={"spark.sql.files.maxPartitionBytes": "33554432"},
        execution_timeout=timedelta(minutes=90),
    )

    # 3-4. Chuẩn bị dataset Kaggle (chạy trong container Airflow, ghi vào ./data mount)
    stage_parquet = PythonOperator(
        task_id="stage_parquet_local",
        python_callable=stage_parquet_from_minio,
        execution_timeout=timedelta(minutes=45),
    )

    export_datasets = PythonOperator(
        task_id="export_kaggle_datasets",
        python_callable=export_kaggle_datasets,
        execution_timeout=timedelta(minutes=120),
    )

    # 5. Push train: retries=0 CỐ Ý — chạy lại sẽ khởi động lần train thứ hai trên GPU
    push_kaggle = PythonOperator(
        task_id="push_training",
        python_callable=push_training,
        retries=0,
        execution_timeout=timedelta(hours=3),
    )

    # 6. Chờ + đăng ký: chạy lại được (--wait-only không push), timeout 14h > 12h của Kaggle
    wait_register = PythonOperator(
        task_id="wait_and_register",
        python_callable=wait_and_register,
        retries=1,
        execution_timeout=timedelta(hours=14),
    )

    verify_alias = PythonOperator(
        task_id="verify_production",
        python_callable=verify_production,
        execution_timeout=timedelta(minutes=15),
    )

    ingest_full >> precompute_masks >> stage_parquet >> export_datasets \
        >> push_kaggle >> wait_register >> verify_alias
