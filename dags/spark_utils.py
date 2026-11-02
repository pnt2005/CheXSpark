"""Cấu hình dùng chung cho các DAG Spark của CheXSpark.

Mọi job đều nộp lên Spark standalone cluster (không chạy local trong Airflow):
  - driver  : chạy trong container Airflow (client deploy mode)
  - executor: chạy trên spark-worker-1/2

Cấu hình S3A/MinIO lặp lại y hệt `infra/spark/conf/spark-defaults.conf`, nhưng
phải truyền qua `--conf` vì driver không đọc file conf của cluster.
Credentials lấy từ env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) — không
hard-code, đúng quy ước của repo.
"""

from __future__ import annotations

import os

from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

APP_DIR = os.getenv("CHEXSPARK_APP_DIR", "/opt/airflow/src")
JAR_DIR = os.getenv("CHEXSPARK_JAR_DIR", "/opt/airflow/jars")

# S3A jars: driver trong container Airflow không có sẵn → nạp qua
# `spark.driver.extraClassPath` (CHỈ driver). Không dùng `--jars` vì Spark sẽ
# phân phối thêm 1 bản hadoop-aws/aws-sdk xuống executor — executor trong image
# cluster đã có sẵn 2 jar này, nạp trùng gây lỗi commit file trên S3A
# ("No such file or directory" khi rename file tạm).
DRIVER_JARS = f"{JAR_DIR}/hadoop-aws-3.3.4.jar:{JAR_DIR}/aws-java-sdk-bundle-1.11.1021.jar"


def spark_conf(driver_memory: str = "1g", executor_memory: str = "1g") -> dict[str, str]:
    """Spark conf cho driver chạy trong Airflow + S3A trỏ MinIO.

    Lưu ý RAM: Docker VM chỉ có 8GB, đang chạy sẵn MinIO/Postgres/MLflow/Airflow
    (~2GB) + 2 worker. Nếu để mặc định, Spark sẽ xin 4 executor × 2GB = 8GB →
    kernel OOM-kill executor giữa job (triệu chứng: `EOFException` khi lấy task
    result). Vì vậy giới hạn `spark.cores.max=2` (2 executor × 1GB).
    """
    return {
        # Driver chạy trong container scheduler → phải khai báo hostname để
        # executor kết nối ngược lại được (cùng docker network chexspark-net)
        "spark.driver.host": os.getenv("SPARK_DRIVER_HOST", "airflow-scheduler"),
        "spark.driver.bindAddress": "0.0.0.0",
        # Nạp S3A jars cho RIÊNG driver (xem chú thích DRIVER_JARS ở đầu file)
        "spark.driver.extraClassPath": DRIVER_JARS,
        "spark.driver.memory": driver_memory,
        "spark.executor.memory": executor_memory,
        "spark.executor.cores": "1",
        "spark.cores.max": os.getenv("SPARK_CORES_MAX", "2"),
        "spark.sql.shuffle.partitions": "8",
        "spark.default.parallelism": "4",
        # S3A / MinIO (giống spark-defaults.conf của cluster)
        "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.aws.credentials.provider":
            "com.amazonaws.auth.EnvironmentVariableCredentialsProvider",
        "spark.hadoop.fs.s3a.change.detection.mode": "none",
        # ── Output committer ─────────────────────────────────────────────────
        # Mặc định (algorithm v1) gom output vào `_temporary/0/task_<job>_m_XXXXX`
        # — tên thư mục này LẶP LẠI giữa các lần chạy (cùng job id), nên nếu một
        # lần chạy lỗi giữa chừng, thư mục tạm cũ sẽ làm lần chạy sau commit hỏng
        # ("No such file or directory" khi rename).
        # v2 commit thẳng task output sang thư mục đích, không dùng chung
        # `_temporary/0` → chạy lại nhiều lần vẫn ổn định.
        # (Không dùng S3A magic/directory committer: Spark 3.5.1 không ship
        #  `spark-hadoop-cloud` nên thiếu PathOutputCommitProtocol.)
        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": "2",
        # KHÔNG set spark.serializer=KryoSerializer: Kryo của driver (pyspark pip)
        # và executor (image cluster) không tương thích → driver không đọc nổi
        # task result (java.io.EOFException trong KryoDeserializationStream).
        # Dùng mặc định JavaSerializer của cluster.
    }


def spark_job(task_id: str, script: str, **kwargs) -> SparkSubmitOperator:
    """Tạo SparkSubmitOperator cho một script trong src/.

    Master KHÔNG truyền trực tiếp: SparkSubmitOperator (provider 4.8.x) lấy
    `--master` từ Airflow connection `spark_default` (host = spark://spark-master:7077),
    khai báo qua env `AIRFLOW_CONN_SPARK_DEFAULT` trong docker-compose.

    Args:
        task_id: ID task trong DAG
        script: đường dẫn tương đối trong src/ (vd "gold/disease_graph.py")
    """
    return SparkSubmitOperator(
        task_id=task_id,
        application=f"{APP_DIR}/{script}",
        conn_id="spark_default",
        name=f"chexspark-{task_id}",
        deploy_mode="client",
        conf=spark_conf(),
        env_vars={"PYTHONPATH": APP_DIR},
        verbose=True,
        **kwargs,
    )
