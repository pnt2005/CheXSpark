"""Giai đoạn DEV — Gold layer: Spark MLlib trên feature vectors DenseNet121 (1024-d).

Đọc `gold/features/densenet121` (Parquet: feat_0000..feat_1023 + 14 label_* + split)
rồi chạy:

1. **PCA** — giải thích phương sai (k=50) + chiếu 2D để trực quan hóa cụm bệnh
2. **K-Means** — quét k=2..8, chọn k theo silhouette, ghi nhãn cụm
3. **Baseline cổ điển** — Logistic Regression + Random Forest cho từng bệnh
   (so sánh với Deep Learning DenseNet121), đánh giá AUC-ROC trên tập test

Output Gold:
  - gold/mllib/pca_variance/    (pc, eigenvalue, explained_variance, cumulative)
  - gold/mllib/pca_points/      (image_index, split, pc1, pc2, label_*, cluster)
  - gold/mllib/kmeans_sweep/    (k, silhouette, wssse)
  - gold/mllib/kmeans_clusters/ (cluster, n_images, top diseases)
  - gold/mllib/baseline_metrics/(disease, lr_auc, rf_auc)

Usage (trong container spark-master):
    /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 --deploy-mode client \\
        --driver-memory 2g --conf spark.executor.memory=2g \\
        /opt/spark/user-src/gold/mllib_analysis.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402
from ingestion.ingest_bronze_to_silver import DISEASES  # noqa: E402

GOLD_FEATURES = os.getenv("GOLD_FEATURES", "s3a://gold/features/densenet121")
GOLD_ML_PCA_VAR = os.getenv("GOLD_ML_PCA_VAR", "s3a://gold/mllib/pca_variance")
GOLD_ML_PCA_POINTS = os.getenv("GOLD_ML_PCA_POINTS", "s3a://gold/mllib/pca_points")
GOLD_ML_KM_SWEEP = os.getenv("GOLD_ML_KM_SWEEP", "s3a://gold/mllib/kmeans_sweep")
GOLD_ML_KM_CLUSTERS = os.getenv("GOLD_ML_KM_CLUSTERS", "s3a://gold/mllib/kmeans_clusters")
GOLD_ML_BASELINE = os.getenv("GOLD_ML_BASELINE", "s3a://gold/mllib/baseline_metrics")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "chexspark-gold-mllib")

K_RANGE = [2, 3, 4, 5, 6, 7, 8]
PCA_K = 50
FEAT_COLS = [f"feat_{i:04d}" for i in range(1024)]


def prepare(spark):
    """Đọc features + assemble vector + standardize."""
    df = spark.read.parquet(GOLD_FEATURES)
    assembler = VectorAssembler(inputCols=FEAT_COLS, outputCol="raw_features")
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withMean=True, withStd=True
    )
    pipeline = Pipeline(stages=[assembler, scaler])
    prepared = pipeline.fit(df).transform(df).cache()
    print(f"Features: {prepared.count()} rows × {len(FEAT_COLS)} dims")
    return prepared


def run_pca(prepared):
    """PCA 50 chiều: phương sai giải thích + chiếu 2D cho scatter plot."""
    pca = PCA(k=PCA_K, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(prepared)

    # Spark PCA: `explainedVariance[i]` ĐÃ là tỉ lệ phương sai (tổng trên toàn bộ
    # 1024 thành phần = 1). Đã kiểm chứng: share × 1024 = phương sai thực của PC.
    # (StandardScaler withStd=True → tổng phương sai toàn không gian = 1024.)
    eig = [float(v) for v in pca_model.explainedVariance.toArray()]
    total_variance = float(len(FEAT_COLS))
    rows, cum = [], 0.0
    for i, share in enumerate(eig):
        cum += share
        rows.append((i, share * total_variance, float(share), float(cum)))
    var_df = prepared.sparkSession.createDataFrame(
        rows, "pc int, eigenvalue double, explained_variance double, cumulative double"
    )

    projected = pca_model.transform(prepared)
    pca2 = projected.select(
        "image_index", "split", *[f"label_{d}" for d in DISEASES],
        vector_to_array("pca_features")[0].alias("pc1"),
        vector_to_array("pca_features")[1].alias("pc2"),
    )
    return pca_model, var_df, pca2, cum


def run_kmeans(prepared, pca2):
    """Quét k, chọn theo silhouette, ghi nhãn cụm + thành phần bệnh của cụm."""
    evaluator = ClusteringEvaluator(
        featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean"
    )
    sweep_rows, best = [], None
    for k in K_RANGE:
        km = KMeans(featuresCol="features", k=k, seed=42, maxIter=50)
        model = km.fit(prepared)
        pred = model.transform(prepared)
        sil = evaluator.evaluate(pred)
        wssse = float(model.summary.trainingCost)
        sweep_rows.append((k, sil, wssse))
        print(f"  [kmeans] k={k}: silhouette={sil:.4f} wssse={wssse:.1f}")
        if best is None or sil > best[1]:
            best = (k, sil, model)

    best_k, best_sil, best_model = best
    print(f"  [kmeans] chọn k={best_k} (silhouette={best_sil:.4f})")

    pred = best_model.transform(prepared).select(
        "image_index", "prediction", *[f"label_{d}" for d in DISEASES]
    ).withColumn(
        "no_finding",
        F.when(F.greatest(*[F.col(f"label_{d}") for d in DISEASES]) == 0, 1).otherwise(0),
    )

    n_total = prepared.count()
    # Thành phần bệnh theo cụm: tỷ lệ từng bệnh trong cụm (so với toàn tập)
    total_by_disease = {
        r["disease"]: r["n"]
        for r in prepared.select(
            F.explode(F.array(*[
                F.struct(F.lit(d).alias("disease"), F.col(f"label_{d}").alias("v"))
                for d in DISEASES
            ])).alias("s")
        ).groupBy("s.disease").agg(F.sum("s.v").alias("n")).collect()
    }
    cluster_rows = []
    for r in pred.groupBy("prediction").count().orderBy("prediction").collect():
        c, n = r["prediction"], r["count"]
        stats = pred.filter(F.col("prediction") == c).select(
            *[F.avg(F.col(f"label_{d}")).alias(d) for d in DISEASES],
            F.avg("no_finding").alias("no_finding"),
        ).collect()[0].asDict()
        # enrichment = tỷ lệ trong cụm / tỷ lệ toàn tập
        enriched = {
            d: (stats[d] or 0.0) / (total_by_disease[d] / n_total)
            for d in DISEASES
            if total_by_disease[d]
        }
        top = sorted(enriched.items(), key=lambda kv: kv[1], reverse=True)[:3]
        cluster_rows.append((
            int(c), int(n),
            ", ".join(f"{d}(x{v:.1f})" for d, v in top),
            float(stats["no_finding"] or 0.0),
        ))

    return sweep_rows, cluster_rows, best_k, best_sil, pred


def run_baselines(prepared, spark):
    """Logistic Regression + Random Forest cho từng bệnh — so sánh với DenseNet121."""
    train = prepared.filter(F.col("split") == "train")
    test = prepared.filter(F.col("split") == "test")
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    rows = []
    for d in DISEASES:
        tr = train.withColumn("label", F.col(f"label_{d}").cast("double"))
        te = test.withColumn("label", F.col(f"label_{d}").cast("double"))
        n_pos = tr.agg(F.sum("label")).collect()[0][0] or 0
        te_pos = te.agg(F.sum("label")).collect()[0][0] or 0
        if n_pos == 0 or n_pos == tr.count():
            print(f"  [baseline] {d}: bỏ qua (nhãn degenerate trong train)")
            rows.append((d, None, None, int(n_pos)))
            continue
        if te_pos == 0 or te_pos == te.count():
            print(f"  [baseline] {d}: bỏ qua (nhãn degenerate trong test)")
            rows.append((d, None, None, int(n_pos)))
            continue

        lr = LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=50, regParam=0.01, elasticNetParam=0.0,
        )
        lr_auc = evaluator.evaluate(lr.fit(tr).transform(te))

        rf = RandomForestClassifier(
            featuresCol="features", labelCol="label",
            numTrees=60, maxDepth=5, seed=42, subsamplingRate=0.8,
        )
        rf_auc = evaluator.evaluate(rf.fit(tr).transform(te))

        print(f"  [baseline] {d}: LR AUC={lr_auc:.4f} | RF AUC={rf_auc:.4f} (train pos={int(n_pos)})")
        rows.append((d, float(lr_auc), float(rf_auc), int(n_pos)))

    schema = "disease string, lr_auc double, rf_auc double, train_positives long"
    return spark.createDataFrame(rows, schema)


def main() -> None:
    spark = get_spark_session("chexspark-gold-mllib")
    spark.sparkContext.setLogLevel("WARN")

    prepared = prepare(spark)

    # --- PCA ---
    pca_model, var_df, pca2, cum = run_pca(prepared)
    print(f"=== PCA: {PCA_K} thành phần giải thích {cum:.1%} phương sai "
          f"(2 PC đầu: {sum(r[2] for r in var_df.collect()[:2]):.1%}) ===")

    # --- K-Means ---
    sweep_rows, cluster_rows, best_k, best_sil, pred = run_kmeans(prepared, pca2)
    print("=== K-MEANS CLUSTERS ===")
    for c in cluster_rows:
        print(f"  cluster {c[0]}: {c[1]} ảnh | top bệnh: {c[2]}")

    # Ghép nhãn cụm vào điểm PCA 2D (trực quan hóa)
    pca_points = pca2.join(pred.select("image_index", F.col("prediction").alias("cluster")),
                           on="image_index", how="left")

    # --- Baseline LR / RF ---
    print("=== BASELINE MLlib (test AUC) ===")
    baseline_df = run_baselines(prepared, spark)

    # --- Ghi Gold ---
    var_df.write.mode("overwrite").parquet(GOLD_ML_PCA_VAR)
    pca_points.write.mode("overwrite").parquet(GOLD_ML_PCA_POINTS)
    sweep_df = spark.createDataFrame(sweep_rows, "k int, silhouette double, wssse double")
    sweep_df.write.mode("overwrite").parquet(GOLD_ML_KM_SWEEP)
    clusters_df = spark.createDataFrame(
        cluster_rows, "cluster int, n_images long, top_diseases string, no_finding_rate double"
    )
    clusters_df.write.mode("overwrite").parquet(GOLD_ML_KM_CLUSTERS)
    baseline_df.write.mode("overwrite").parquet(GOLD_ML_BASELINE)

    # --- MLflow ---
    lr_aucs = [r["lr_auc"] for r in baseline_df.collect() if r["lr_auc"] is not None]
    rf_aucs = [r["rf_auc"] for r in baseline_df.collect() if r["rf_auc"] is not None]
    log_to_mlflow({
        "mllib_pca_components": PCA_K,
        "mllib_pca_cum_variance": cum,
        "mllib_kmeans_k": best_k,
        "mllib_kmeans_silhouette": best_sil,
        "mllib_lr_mean_auc": float(sum(lr_aucs) / len(lr_aucs)) if lr_aucs else 0.0,
        "mllib_rf_mean_auc": float(sum(rf_aucs) / len(rf_aucs)) if rf_aucs else 0.0,
        "mllib_n_labels_evaluated": len(lr_aucs),
    })

    spark.stop()


def log_to_mlflow(metrics: dict) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"gold-mllib-{datetime.utcnow():%Y%m%d-%H%M%S}"):
            mlflow.log_params({
                "layer": "gold",
                "analysis": "mllib_pca_kmeans_lr_rf",
                "feature_source": "DenseNet121 1024-d (global avg pool, Stage-2 checkpoint)",
                "pca_k": PCA_K,
                "kmeans_k_range": ",".join(str(k) for k in K_RANGE),
                "models": "LogisticRegression, RandomForest(numTrees=60,maxDepth=5)",
            })
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            print(f"Logged run to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
