"""Kiểm tra nhanh Silver layer: đọc 1 partition, in schema + vài dòng.

Usage trong container spark-master:
    /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
        /opt/spark/user-src/ingestion/verify_silver.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402

SILVER_PATH = "s3a://silver/cxr14"


def main() -> None:
    spark = get_spark_session("verify-silver")
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(SILVER_PATH)
    print("=== SCHEMA ===")
    df.printSchema()
    print("=== TOTAL ROWS ===", df.count())
    print("=== PER-SPLIT COUNTS ===")
    df.groupBy("split").count().orderBy("split").show()
    print("=== SAMPLE (image, patient, split, n_diseases, labels) ===")
    label_cols = [c for c in df.columns if c.startswith("label_")]
    df.select("image_index", "patient_id", "split", "n_diseases", *label_cols[:4]).show(5, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
