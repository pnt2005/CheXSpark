"""Giai đoạn DEV — Gold layer: Disease Co-occurrence + Thống kê dịch tễ.

Đọc FULL metadata NIH (~112k dòng) từ Bronze, tính:
1. Ma trận đồng xuất hiện 14×14 (co-occurrence) + Jaccard + xác suất có điều kiện
2. Thống kê dịch tễ: phân bố bệnh, comorbidity, theo giới/tuổi/view

Ghi Gold layer (Parquet) phục vụ dashboard + báo cáo:
  - gold/cooccurrence/pairs/    (d1, d2, count, jaccard, p_d2_given_d1, ...)
  - gold/cooccurrence/matrix/   (ma trận 14×14 dạng long: d1, d2, count)
  - gold/stats/disease/         (mỗi bệnh: count, prevalence, theo giới/view)
  - gold/stats/comorbidity/     (phân bố số bệnh/ảnh)
  - gold/stats/top_pairs/       (top cặp đồng mắc)

Usage (trong container spark-master):
    /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 --deploy-mode client \
        /opt/spark/user-src/gold/disease_cooccurrence.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402
from ingestion.ingest_bronze_to_silver import (  # noqa: E402
    DISEASES,
    load_metadata,
    with_multihot_labels,
)

GOLD_COOCC_PAIRS = "s3a://gold/cooccurrence/pairs"
GOLD_COOCC_MATRIX = "s3a://gold/cooccurrence/matrix"
GOLD_STATS_DISEASE = "s3a://gold/stats/disease"
GOLD_STATS_COMORBIDITY = "s3a://gold/stats/comorbidity"
GOLD_STATS_TOP_PAIRS = "s3a://gold/stats/top_pairs"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "chexspark-gold-cooccurrence")


def explode_positive_labels(df):
    """Tạo bảng (image_index, disease) chỉ cho các nhãn = 1."""
    disease_array = F.array(
        *[F.when(F.col(f"label_{d}") == 1, F.lit(d)) for d in DISEASES]
    )
    return (
        df.select("image_index", F.explode(disease_array).alias("disease"))
        .filter(F.col("disease").isNotNull())
    )


def compute_pairs(exploded):
    """Self-join để đếm cặp đồng xuất hiện (d1 < d2, không trùng, không tự cặp)."""
    a = exploded.alias("a")
    b = exploded.alias("b")
    return (
        a.join(b, F.col("a.image_index") == F.col("b.image_index"))
        .filter(F.col("a.disease") < F.col("b.disease"))
        .groupBy(
            F.col("a.disease").alias("disease_1"),
            F.col("b.disease").alias("disease_2"),
        )
        .agg(F.count(F.lit(1)).alias("cooccurrence"))
    )


def enrich_pairs(pairs, disease_counts):
    """Thêm Jaccard + xác suất có điều kiện 2 chiều vào từng cặp."""
    c1 = disease_counts.select(
        F.col("disease").alias("disease_1"), F.col("count").alias("count_1")
    )
    c2 = disease_counts.select(
        F.col("disease").alias("disease_2"), F.col("count").alias("count_2")
    )
    enriched = pairs.join(c1, on="disease_1").join(c2, on="disease_2")
    union = F.col("count_1") + F.col("count_2") - F.col("cooccurrence")
    return enriched.select(
        "disease_1", "disease_2", "cooccurrence", "count_1", "count_2",
        (F.col("cooccurrence") / union).alias("jaccard"),
        (F.col("cooccurrence") / F.col("count_1")).alias("p_d2_given_d1"),
        (F.col("cooccurrence") / F.col("count_2")).alias("p_d1_given_d2"),
    )


def build_matrix(enriched_pairs):
    """Ma trận 14×14 đầy đủ (long form): cả cặp có count=0, đường chéo = count bệnh."""
    pairs_all = []
    for i, d1 in enumerate(DISEASES):
        for d2 in DISEASES[i + 1 :]:
            pairs_all.append((d1, d2))
    schema = "disease_1 string, disease_2 string"
    all_pairs = enriched_pairs.sparkSession.createDataFrame(pairs_all, schema)
    filled = (
        all_pairs.join(
            enriched_pairs.select("disease_1", "disease_2", "cooccurrence"),
            on=["disease_1", "disease_2"],
            how="left",
        )
        .fillna(0, subset=["cooccurrence"])
    )
    # Thêm cặp đảo (đối xứng) để tiện vẽ heatmap
    reversed_pairs = filled.select(
        F.col("disease_2").alias("disease_1"),
        F.col("disease_1").alias("disease_2"),
        "cooccurrence",
    )
    return filled.union(reversed_pairs)


def compute_disease_stats(df):
    """Thống kê mỗi bệnh: tổng ca, prevalence, theo giới, theo view.

    Gộp 14 bệnh × 5 metric vào **một** lần agg (1 pass trên full 112k thay vì
    14 scan riêng). Boolean phải cast int vì Spark `sum()` không nhận BOOLEAN.
    """
    total = df.count()

    def _cond(col):
        return col.cast("int")

    agg_exprs = []
    for d in DISEASES:
        col = F.col(f"label_{d}")
        agg_exprs += [
            F.sum(_cond(col)).alias(f"cases_{d}"),
            F.sum(_cond((col == 1) & (F.col("patient_gender") == "M"))).alias(f"male_{d}"),
            F.sum(_cond((col == 1) & (F.col("patient_gender") == "F"))).alias(f"female_{d}"),
            F.sum(_cond((col == 1) & (F.col("view_position") == "PA"))).alias(f"pa_{d}"),
            F.sum(_cond((col == 1) & (F.col("view_position") == "AP"))).alias(f"ap_{d}"),
        ]
    agg = df.agg(*agg_exprs).collect()[0].asDict()

    rows = []
    for d in DISEASES:
        cases = agg[f"cases_{d}"] or 0
        rows.append((
            d,
            cases,
            cases / total if total else 0.0,
            agg[f"male_{d}"] or 0,
            agg[f"female_{d}"] or 0,
            agg[f"pa_{d}"] or 0,
            agg[f"ap_{d}"] or 0,
        ))
    schema = (
        "disease string, cases long, prevalence double, "
        "male long, female long, pa long, ap long"
    )
    return df.sparkSession.createDataFrame(rows, schema)


def compute_comorbidity(df):
    """Phân bố số bệnh/ảnh (0 = No Finding, 1..14)."""
    return (
        df.groupBy("n_diseases")
        .agg(F.count(F.lit(1)).alias("images"))
        .orderBy("n_diseases")
    )


def main() -> None:
    spark = get_spark_session("chexspark-gold-cooccurrence")
    spark.sparkContext.setLogLevel("WARN")

    # 1) Full metadata 112k + multi-hot labels
    metadata = with_multihot_labels(load_metadata(spark)).cache()
    total = metadata.count()
    print(f"Full metadata rows: {total}")

    # 2) Đếm từng bệnh (dùng cho Jaccard/conditional prob)
    disease_counts = metadata.agg(
        *[F.sum(F.col(f"label_{d}")).alias(d) for d in DISEASES]
    ).select(
        F.explode(F.array(*[
            F.struct(F.lit(d).alias("disease"), F.col(d).cast("long").alias("count"))
            for d in DISEASES
        ])).alias("s")
    ).select("s.disease", "s.count")

    # 3) Cặp đồng xuất hiện + enrich
    exploded = explode_positive_labels(metadata).cache()
    pairs = compute_pairs(exploded)
    enriched = enrich_pairs(pairs, disease_counts).cache()

    # 4) Ma trận 14×14 + comorbidity
    matrix = build_matrix(enriched)
    comorbidity = compute_comorbidity(metadata)
    disease_stats = compute_disease_stats(metadata)
    top_pairs = enriched.orderBy(F.desc("cooccurrence")).limit(20)

    # 5) Ghi Gold
    (enriched.write.mode("overwrite").parquet(GOLD_COOCC_PAIRS))
    (matrix.write.mode("overwrite").parquet(GOLD_COOCC_MATRIX))
    (disease_stats.write.mode("overwrite").parquet(GOLD_STATS_DISEASE))
    (comorbidity.write.mode("overwrite").parquet(GOLD_STATS_COMORBIDITY))
    (top_pairs.write.mode("overwrite").parquet(GOLD_STATS_TOP_PAIRS))

    # 6) In kết quả chính
    print("=== DISEASE COUNTS (full 112k) ===")
    for row in disease_counts.orderBy(F.desc("count")).collect():
        print(f"  {row['disease']}: {row['count']}")
    print("=== TOP 10 CO-OCCURRENCE PAIRS ===")
    for row in top_pairs.limit(10).collect():
        print(
            f"  {row['disease_1']} + {row['disease_2']}: {row['cooccurrence']} "
            f"(Jaccard {row['jaccard']:.3f})"
        )
    print("=== COMORBIDITY DISTRIBUTION ===")
    for row in comorbidity.collect():
        print(f"  n_diseases={row['n_diseases']}: {row['images']}")

    # 7) MLflow
    metrics = {
        "gold_total_images": total,
        "gold_pairs_nonzero": enriched.count(),
        "gold_comorbidity_ge2": metadata.filter(F.col("n_diseases") >= 2).count(),
    }
    log_to_mlflow(metrics)

    spark.stop()


def log_to_mlflow(metrics: dict) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"gold-cooccurrence-{datetime.utcnow():%Y%m%d-%H%M%S}"):
            mlflow.log_params({
                "layer": "gold",
                "analysis": "disease_cooccurrence",
                "source": "s3a://bronze/metadata/Data_Entry_2017_v2020.csv",
                "n_diseases": len(DISEASES),
                "outputs": "cooccurrence/pairs, cooccurrence/matrix, stats/*",
            })
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            print(f"Logged run to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
