"""CheXSpark Spark session factory.

Mọi job Spark (ingestion, silver, gold) đều dùng module này để tạo session
tương thích với cluster standalone trong docker-compose.yml và truy cập MinIO.
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark_session(app_name: str) -> SparkSession:
    """Tạo SparkSession trỏ tới spark://spark-master:7077 (hoặc local nếu không có cluster).

    Credentials MinIO đọc từ env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    (đã được inject vào container bởi docker-compose).
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("SPARK_S3_ENDPOINT", "http://minio:9000"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
    )

    master = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    if os.getenv("SPARK_LOCAL", "0") == "1":
        builder = builder.master("local[*]")
    else:
        builder = builder.master(master)

    return builder.getOrCreate()
