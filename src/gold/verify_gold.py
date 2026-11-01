"""Verify Gold layer: đọc lại các output co-occurrence/graph/stats, in tóm tắt.

Usage (trong container spark-master):
    /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
        --deploy-mode client /opt/spark/user-src/gold/verify_gold.py
"""

import os
import sys

from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402

GOLD_DATASETS = {
    "cooccurrence/pairs": "s3a://gold/cooccurrence/pairs",
    "cooccurrence/matrix": "s3a://gold/cooccurrence/matrix",
    "stats/disease": "s3a://gold/stats/disease",
    "stats/comorbidity": "s3a://gold/stats/comorbidity",
    "stats/top_pairs": "s3a://gold/stats/top_pairs",
    "stats/top_associations": "s3a://gold/stats/top_associations",
    "graph/nodes": "s3a://gold/graph/nodes",
    "graph/edges": "s3a://gold/graph/edges",
    "graph/edges_all": "s3a://gold/graph/edges_all",
    "graph/threshold_sweep": "s3a://gold/graph/threshold_sweep",
    "graph/summary": "s3a://gold/graph/summary",
    "mllib/pca_variance": "s3a://gold/mllib/pca_variance",
    "mllib/kmeans_sweep": "s3a://gold/mllib/kmeans_sweep",
    "mllib/kmeans_clusters": "s3a://gold/mllib/kmeans_clusters",
    "mllib/baseline_metrics": "s3a://gold/mllib/baseline_metrics",
}


def main() -> None:
    spark = get_spark_session("verify-gold")
    spark.sparkContext.setLogLevel("ERROR")

    for name, path in GOLD_DATASETS.items():
        df = spark.read.parquet(path)
        print(f"=== {name} ({df.count()} rows) ===")
        df.printSchema()
        df.show(5, truncate=False)

    print("=== TOP 5 CENTRAL DISEASES ===")
    spark.read.parquet(GOLD_DATASETS["graph/nodes"]).orderBy(
        F.desc("weighted_degree")
    ).show(5, truncate=False)

    print("=== TOP 10 CO-OCCURRENCE PAIRS ===")
    spark.read.parquet(GOLD_DATASETS["stats/top_pairs"]).orderBy(
        F.desc("cooccurrence")
    ).show(10, truncate=False)

    print("=== GRAPH SUMMARY ===")
    spark.read.parquet(GOLD_DATASETS["graph/summary"]).show(truncate=False)

    print("=== PCA VARIANCE (top 5 PC) ===")
    spark.read.parquet(GOLD_DATASETS["mllib/pca_variance"]).orderBy("pc").show(5)

    print("=== KMEANS SWEEP ===")
    spark.read.parquet(GOLD_DATASETS["mllib/kmeans_sweep"]).orderBy("k").show()

    print("=== KMEANS CLUSTERS ===")
    spark.read.parquet(GOLD_DATASETS["mllib/kmeans_clusters"]).orderBy("cluster").show(
        truncate=False
    )

    print("=== BASELINE MLlib (LR / RF) ===")
    spark.read.parquet(GOLD_DATASETS["mllib/baseline_metrics"]).orderBy(
        F.desc("lr_auc")
    ).show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
