"""Giai đoạn DEV — Ingestion: CheXmask (Bronze) → join Silver.

Pipeline Spark đọc mask phổi CheXmask (RLE, dạng CSV ~2GB / 112k rows) từ
Bronze, join với Silver Parquet hiện tại theo Image ID và ghi Silver mới
(`silver/cxr14_masked`) kèm 7 cột mask/quality — đúng tinh thần AGENTS.md:
Silver = "map chuẩn hóa Image ID giữa ảnh và mask".

Điểm mấu chốt "Big Data": CheXmask full ~112k rows được Spark đọc phân tán
từ MinIO, filter/broadcast join với dev subset 2k, chỉ materialize kết quả
cần thiết. Cột Landmarks (chuỗi tọa độ cực lớn, không dùng cho segmentation)
bị drop ngay sau khi đọc.

Output: s3a://silver/cxr14_masked/ (partition theo split) — GHI MỚI thay vì
overwrite cxr14 để không phá dữ liệu cũ nếu job fail giữa chừng.

Usage (trong container spark-master):
    /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 --deploy-mode client \
        /opt/spark/user-src/ingestion/ingest_chexmask_to_silver.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402

BRONZE_CHEXMASK = "s3a://bronze/chexmask/ChestX-Ray8.csv"
SILVER_IN = "s3a://silver/cxr14"
SILVER_OUT = "s3a://silver/cxr14_masked"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "chexspark-silver-mask-join")

# Header CheXmask: Image Index, Dice RCA (Mean), Dice RCA (Max), Landmarks,
# Left Lung, Right Lung, Heart, Height, Width.
# Landmarks là JSON/CSV array hàng nghìn tọa độ — đọc tường minh rồi DROP ngay
# để không tràn memory và không ảnh hưởng schema output.
CHEXMASK_SCHEMA = StructType([
    StructField("image_index", StringType()),
    StructField("dice_rca_mean", DoubleType()),
    StructField("dice_rca_max", DoubleType()),
    StructField("landmarks", StringType()),   # drop ngay sau khi đọc
    StructField("mask_left_rle", StringType()),
    StructField("mask_right_rle", StringType()),
    StructField("mask_heart_rle", StringType()),
    StructField("mask_height", IntegerType()),
    StructField("mask_width", IntegerType()),
])

# Cột mask/quality được thêm vào Silver
MASK_COLS = [
    "dice_rca_mean", "dice_rca_max",
    "mask_left_rle", "mask_right_rle", "mask_heart_rle",
    "mask_height", "mask_width",
]


def load_chexmask(spark):
    """Đọc full CheXmask (~112k rows, RLE strings) từ MinIO Bronze, drop Landmarks."""
    return (
        spark.read
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .schema(CHEXMASK_SCHEMA)
        .csv(BRONZE_CHEXMASK)
        .drop("landmarks")
    )


def load_silver(spark):
    """Đọc Silver Parquet hiện tại (labels + split, dev subset)."""
    return spark.read.parquet(SILVER_IN)


def filter_masks_to_silver(masks_df, silver_df):
    """Giảm CheXmask 112k -> chỉ rows khớp dev subset, SOM NHAT co the.

    RLE strings khong duoc phep di xuyen qua shuffle/cache khi con 112k rows
    (gây OOM trên cluster 8GB) — filter inner ngay bang broadcast(silver_ids),
    downstream chi con 2k rows (~36MB) an toan.
    """
    silver_ids = silver_df.select("image_index")
    return masks_df.join(F.broadcast(silver_ids), on="image_index", how="inner")


def validate_joined(df):
    """Kiểm tra chất lượng join: row count, null RLE, phân bố Dice RCA."""
    stats = df.agg(
        F.count(F.lit(1)).alias("masked_total_images"),
        F.sum(F.col("mask_left_rle").isNull().cast("int")).alias("missing_left_rle"),
        F.sum(F.col("mask_right_rle").isNull().cast("int")).alias("missing_right_rle"),
        F.sum((F.col("mask_height") != 1024).cast("int")).alias("non_1024_masks"),
        F.avg("dice_rca_mean").alias("mean_dice_rca"),
        F.expr("percentile_approx(dice_rca_mean, 0.5)").alias("median_dice_rca"),
        F.sum((F.col("dice_rca_mean") >= 0.7).cast("int")).alias("dice_ge_0_7"),
        F.sum((F.col("split") == "train").cast("int")).alias("masked_train"),
        F.sum((F.col("split") == "val").cast("int")).alias("masked_val"),
        F.sum((F.col("split") == "test").cast("int")).alias("masked_test"),
    ).collect()[0].asDict()
    return stats


def main() -> None:
    spark = get_spark_session("chexspark-silver-mask-join")
    spark.sparkContext.setLogLevel("WARN")

    # 1) Silver hiện tại + CheXmask full từ Bronze
    silver = load_silver(spark)
    silver_count = silver.count()
    masks = load_chexmask(spark)
    print(f"Silver rows: {silver_count}")

    # 2) Filter CheXmask ve dev subset NGAY (112k -> ~2k) truoc moi
    #    cache/agg — RLE strings qua shuffle 112k rows gay OOM tren 8GB
    matched_masks = filter_masks_to_silver(masks, silver)
    matched_count = matched_masks.count()
    missing_count = silver_count - matched_count
    print(f"CheXmask khop dev subset: {matched_count}/{silver_count} "
          f"(thieu mask: {missing_count})")

    # 3) Join Silver voi masks da filter + timestamp
    joined = silver.join(
        F.broadcast(matched_masks), on="image_index", how="inner"
    ).withColumn(
        "chexmask_ingested_at", F.lit(datetime.utcnow().isoformat())
    ).cache()
    print(f"Ảnh dev thiếu mask: {missing_count}")

    # 4) Validate chất lượng trước khi ghi
    stats = validate_joined(joined)
    stats["matched_mask_rows"] = matched_count
    stats["missing_mask_images"] = missing_count
    print("=== MASK JOIN STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["masked_total_images"] != matched_count:
        raise RuntimeError("Row count sau join khong khop — dung pipeline!")

    # 5) Ghi Silver mới (partition theo split) — KHÔNG overwrite cxr14 cũ
    (joined.write.mode("overwrite")
     .partitionBy("split")
     .parquet(SILVER_OUT))
    print(f"Đã ghi {stats['masked_total_images']} rows -> {SILVER_OUT}")

    # 6) Log MLflow
    log_to_mlflow(stats)

    spark.stop()


def log_to_mlflow(stats: dict) -> None:
    """Ghi params/metrics của mask join lên MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"mask-join-{datetime.utcnow():%Y%m%d-%H%M%S}"):
            mlflow.log_params({
                "mask_source": BRONZE_CHEXMASK,
                "mask_format": "rle_start_length_pairs_1024",
                "silver_input": SILVER_IN,
                "silver_output": SILVER_OUT,
                "join_key": "image_index",
                "dice_threshold": 0.7,
            })
            mlflow.log_metrics({k: float(v) for k, v in stats.items()})
            print(f"Logged run to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
