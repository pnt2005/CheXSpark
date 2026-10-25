"""Kiểm tra nhanh Silver layer: đọc 1 partition, in schema + vài dòng.

Usage trong container spark-master:
    /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
        /opt/spark/user-src/ingestion/verify_silver.py [SILVER_PATH]

SILVER_PATH mặc định s3a://silver/cxr14, truyền argv[1] để verify bản khác
(ví dụ s3a://silver/cxr14_masked).
"""

import os
import sys

from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402

SILVER_PATH = sys.argv[1] if len(sys.argv) > 1 else "s3a://silver/cxr14"


def main() -> None:
    spark = get_spark_session("verify-silver")
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(SILVER_PATH)
    print("=== SCHEMA ===", SILVER_PATH)
    df.printSchema()
    print("=== TOTAL ROWS ===", df.count())
    print("=== PER-SPLIT COUNTS ===")
    df.groupBy("split").count().orderBy("split").show()
    print("=== SAMPLE (image, patient, split, n_diseases, labels) ===")
    label_cols = [c for c in df.columns if c.startswith("label_")]
    df.select("image_index", "patient_id", "split", "n_diseases", *label_cols[:4]).show(5, truncate=False)
    if "dice_rca_mean" in df.columns:
        print("=== MASK COLS SAMPLE (dice, RLE lens, dims) ===")
        df.select(
            "image_index", "dice_rca_mean",
            F.length("mask_left_rle").alias("left_rle_len"),
            F.length("mask_right_rle").alias("right_rle_len"),
            "mask_height", "mask_width",
        ).show(5, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
