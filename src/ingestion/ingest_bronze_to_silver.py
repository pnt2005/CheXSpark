"""Giai đoạn DEV — Ingestion: Bronze → Silver.

Pipeline Spark (standalone cluster) đọc metadata NIH ChestX-ray14 từ MinIO,
xử lý đa nhãn 14 bệnh, tách tập theo bệnh nhân (patient-level split, tránh
rò rỉ bệnh nhân) và ghi Silver layer dạng Parquet partition theo split.

Điểm mấu chốt "Big Data": job đọc toàn bộ metadata (~112k dòng) phân tán trên
cluster, chỉ Materialize 2k ảnh dev subset để huấn luyện nhanh, nhưng thống kê
dataset tính trên full metadata.

Usage (trong container spark-master):
    /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 --deploy-mode client \
        /opt/spark/user-src/ingestion/ingest_bronze_to_silver.py

    # Toàn bộ 112k dòng (bỏ join manifest) -> silver/cxr14_full:
    # nguồn labels/split cho việc train 112k + precompute_masks --source bronze
    ... ingest_bronze_to_silver.py --full
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

# Cho phép import package chung từ /opt/spark/user-src (mounted trong compose)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402

# 14 bệnh lý lồng ngực theo AGENTS.md / NIH ChestX-ray14
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

BRONZE_METADATA = "s3a://bronze/metadata/Data_Entry_2017_v2020.csv"
BRONZE_MANIFEST = "s3a://bronze/images_manifest/images_manifest.csv"
SILVER_OUT = "s3a://silver/cxr14"
# `--full`: toàn bộ 112k dòng (labels + patient-level split) — dùng làm nguồn
# split/labels cho việc train 112k. Khác `cxr14` (dev 2k, join manifest ảnh).
SILVER_FULL_OUT = "s3a://silver/cxr14_full"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "chexspark-silver-ingestion")

# Header gốc có 9 trường nhưng dòng dữ liệu có 11 giá trị (2 header bao trùm
# 2 cột each: OriginalImage[Width,Height] và PixelSpacing[x,y]).
# Ta đọc không header và gán schema 11 cột tường minh để tránh lệch cột.
METADATA_SCHEMA = StructType([
    StructField("image_index", StringType()),
    StructField("finding_labels", StringType()),
    StructField("followup_no", IntegerType()),
    StructField("patient_id", IntegerType()),
    StructField("patient_age", StringType()),  # có thể có "N/A" -> xử lý sau
    StructField("patient_gender", StringType()),
    StructField("view_position", StringType()),
    StructField("orig_width", IntegerType()),
    StructField("orig_height", IntegerType()),
    StructField("pixel_spacing_x", DoubleType()),
    StructField("pixel_spacing_y", DoubleType()),
])


def load_metadata(spark):
    """Đọc toàn bộ metadata NIH (~112k dòng) từ MinIO Bronze."""
    return (
        spark.read
        .option("header", False)
        .option("mode", "PERMISSIVE")
        .schema(METADATA_SCHEMA)
        .csv(BRONZE_METADATA)
        # Loại dòng header (với header=False, dòng đầu bị đọc thành data;
        # dòng header có patient_id=None do parse 'Patient ID' -> int thất bại)
        .filter(F.col("patient_id").isNotNull())
    )


def load_manifest(spark):
    """Đọc danh sách ảnh dev subset đã materialize trong Bronze."""
    return spark.read.option("header", True).csv(BRONZE_MANIFEST).select("image_index")


def with_multihot_labels(df):
    """Thêm 14 cột binary (0/1) từ cột finding_labels phân cách bằng '|'."""
    split_labels = F.split(F.col("finding_labels"), r"\|")
    for disease in DISEASES:
        df = df.withColumn(
            f"label_{disease}",
            F.array_contains(split_labels, F.lit(disease)).cast("int"),
        )
    # Tổng số bệnh thật trên 1 ảnh (loại 'No Finding') — phục vụ thống kê comorbidity
    df = df.withColumn(
        "n_diseases",
        F.size(F.filter(split_labels, lambda x: x != F.lit("No Finding"))),
    ).withColumn(
        "no_finding",
        F.array_contains(split_labels, F.lit("No Finding")).cast("int"),
    )
    return df


def patient_level_split(df, train_pct=70, val_pct=15):
    """Tách train/val/test theo patient_id (mọi ảnh của 1 bệnh nhân vào 1 tập).

    Dùng hash(patient_id) mod 100 để đảm bảo:
    - Không rò rỉ bệnh nhân giữa các tập (patient leakage).
    - Tỷ lệ ổn định, không cần load toàn bộ dữ liệu vào 1 máy (Spark-friendly).
    """
    bucket = F.pmod(F.hash(F.col("patient_id")), F.lit(100))
    return df.withColumn(
        "split",
        F.when(bucket < F.lit(train_pct), "train")
        .when(bucket < F.lit(train_pct + val_pct), "val")
        .otherwise("test"),
    )


def compute_full_stats(metadata_df):
    """Thống kê trên FULL metadata (~112k) trong 1 pass agg — thể hiện quy mô Big Data."""
    split_labels = F.split(F.col("finding_labels"), r"\|")
    is_disease = lambda d: F.array_contains(split_labels, F.lit(d)).cast("int")  # noqa: E731
    n_real = F.size(F.filter(split_labels, lambda x: x != F.lit("No Finding")))

    agg_exprs = [
        F.count(F.lit(1)).alias("full_total_images"),
        F.countDistinct("patient_id").alias("full_total_patients"),
        F.sum((n_real >= 2).cast("int")).alias("full_comorbidity_ge2"),
        F.sum((n_real == 0).cast("int")).alias("full_no_finding"),
    ] + [F.sum(is_disease(d)).alias(f"full_pos_{d}") for d in DISEASES]

    return metadata_df.agg(*agg_exprs).collect()[0].asDict()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Bronze metadata -> Silver Parquet")
    ap.add_argument("--full", action="store_true",
                    help="materialize toàn bộ 112k dòng (bỏ join manifest ảnh) "
                         "ra silver/cxr14_full — nguồn labels/split cho train 112k")
    ap.add_argument("--out", default=None,
                    help="đường dẫn output Parquet (mặc định: cxr14 hoặc cxr14_full)")
    args = ap.parse_args()
    out = args.out or (SILVER_FULL_OUT if args.full else SILVER_OUT)

    spark = get_spark_session("chexspark-silver-ingestion")
    spark.sparkContext.setLogLevel("WARN")

    # 1) Toàn bộ metadata (~112k dòng) — Spark phân tán trên cluster
    metadata = with_multihot_labels(load_metadata(spark))
    full_stats = compute_full_stats(metadata)

    # 2) Dev subset: inner join metadata với manifest ảnh đã materialize (2k).
    #    `--full` bỏ bước này -> tất cả 112k dòng (ảnh lấy từ Kaggle khi train).
    if args.full:
        subset = metadata
        print("[full] bỏ join manifest -> materialize toàn bộ 112k dòng")
    else:
        manifest = load_manifest(spark)
        subset = metadata.join(F.broadcast(manifest), on="image_index", how="inner")

    # 3) Patient-level split (tránh rò rỉ bệnh nhân) — hash(patient_id) cố định
    #    nên dev 2k và full 112k luôn gán cùng một split cho cùng bệnh nhân.
    subset = patient_level_split(subset)

    # 4) Chuẩn bị cột output Silver
    label_cols = [f"label_{d}" for d in DISEASES]
    silver = subset.select(
        "image_index", "patient_id", "split", "finding_labels",
        "patient_age", "patient_gender", "view_position",
        "n_diseases", "no_finding", *label_cols,
    ).withColumn(
        # URI ảnh trong Bronze để Stage-2 (U-Net/DenseNet) truy cập
        "image_uri", F.concat(F.lit("s3a://bronze/images/"), F.col("image_index"))
    ).withColumn(
        "silver_ingested_at", F.lit(datetime.utcnow().isoformat())
    )

    # Cache để tránh tính lại lineage 2 lần (write + agg stats)
    silver = silver.cache()

    # 5) Ghi Silver Parquet, partition theo split (train/val/test)
    (silver.write.mode("overwrite")
     .partitionBy("split")
     .parquet(out))

    subset_row = silver.agg(
        F.count(F.lit(1)).alias("subset_total_images"),
        F.countDistinct("patient_id").alias("subset_unique_patients"),
        F.sum((F.col("split") == "train").cast("int")).alias("subset_train"),
        F.sum((F.col("split") == "val").cast("int")).alias("subset_val"),
        F.sum((F.col("split") == "test").cast("int")).alias("subset_test"),
    ).collect()[0].asDict()
    subset_stats = subset_row

    print("=== FULL DATASET STATS (112k metadata) ===")
    for k, v in full_stats.items():
        print(f"  {k}: {v}")
    scope = "full" if args.full else "dev"
    print(f"=== {'FULL 112k' if args.full else 'DEV SUBSET (materialized images)'} STATS ===")
    for k, v in subset_stats.items():
        print(f"  {k}: {v}")
    print(f"Output: {out}")

    # 6) Log lên MLflow
    log_to_mlflow(full_stats, subset_stats, out=out, scope=scope)

    spark.stop()


def log_to_mlflow(full_stats: dict, subset_stats: dict, *, out: str,
                  scope: str) -> None:
    """Ghi params/metrics lên MLflow tracking server (http://mlflow:5000)."""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"silver-ingest-{scope}-{datetime.utcnow():%Y%m%d-%H%M%S}"):
            mlflow.log_params({
                "dataset": "NIH ChestX-ray14",
                "metadata_source": BRONZE_METADATA,
                "split_strategy": "patient_level_hash_mod100",
                "train_pct": 70, "val_pct": 15, "test_pct": 15,
                "n_diseases": len(DISEASES),
                "scope": scope,
                "output": out,
            })
            # Log full stats + subset stats dưới dạng metrics
            all_metrics = {**full_stats, **subset_stats}
            mlflow.log_metrics({k: float(v) for k, v in all_metrics.items()})
            print(f"Logged run to MLflow experiment '{MLFLOW_EXPERIMENT}' @ {MLFLOW_TRACKING_URI}")
    except Exception as exc:  # noqa: BLE001
        # Không chặn pipeline nếu MLflow lỗi — in cảnh báo để xử lý sau
        print(f"[WARN] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
