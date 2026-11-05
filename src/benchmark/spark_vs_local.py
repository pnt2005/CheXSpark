"""Benchmark **Single machine vs Spark cluster** — CheXSpark.

So sánh hiệu năng của **cùng một workload** chạy trên 2 engine:

* ``local`` — máy dev đơn lẻ (pandas / PyTorch CPU / scikit-learn, dùng toàn bộ core);
* ``spark``  — cluster standalone trong docker-compose
  (2 worker × 2 core = 4 core, cùng host) — Spark SQL / Spark MLlib /
  PyTorch CPU chạy trong executor.

Workload:

===========  ==================================================================
etl          Đọc Silver metadata 112k (``s3a://silver/cxr14_full``) + aggregate
             (số bệnh, đồng mắc ≥2, split) — scale ×1/×5/×10 để thấy khả năng
             mở rộng
preproc      Decode + resize ảnh PNG từ MinIO → **throughput ảnh/giây**
inference    DenseNet121 production (MLflow registry) forward → **throughput
             ảnh/giây**
train        Huấn luyện DenseNet121 N bước optimizer → **steps/giây**
kmeans       StandardScaler + K-Means(k=10) + PCA(k=2) trên feature 1024-d
             (sklearn vs Spark MLlib)
===========  ==================================================================

Chạy (runner gom số liệu: ``scripts/dev/benchmark_spark_vs_local.py``)::

    # local (máy dev)
    python src/benchmark/spark_vs_local.py --engine local \
        --workloads etl,preproc,inference,train,kmeans

    # spark (trong container master)
    docker compose exec -T spark-master /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        --conf spark.ui.showConsoleProgress=false \
        /opt/spark/user-src/benchmark/spark_vs_local.py --engine spark \
        --workloads etl,preproc,inference,train,kmeans \
        --s3-endpoint http://minio:9000 --mlflow-uri http://mlflow:5000 \
        --torch-threads 2

Mỗi workload in ra đúng một dòng ``RESULT_JSON: {...}`` (không lẫn log khác
trên cùng dòng) để runner phân tích.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# ── sys.path: cho phép `from common...`, `from dl...` ở cả driver lẫn worker ──
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

IMG_SIZE = 224
MLFLOW_MODEL_URI = "models:/chexspark-stage2-densenet121@production"
BUCKET_IMAGES = "bronze"          # bucket chứa ảnh PNG (dev subset 2000)
PREFIX_IMAGES = "images/"
SILVER_FULL = "s3a://silver/cxr14_full"
GOLD_FEATURES = "s3a://gold/features/densenet121"

# Cache model trong process worker (Spark tái sử dụng Python worker giữa các task)
_MODEL_CACHE: dict = {}
_SESSION: dict = {"spark": None, "session_s": None}


def _add_src_path() -> None:
    """Worker Spark (cloudpickle) không có __file__ → tự chèn đường dẫn src."""
    import sys as _sys

    for p in ("/opt/spark/user-src",):          # container Spark (mount ./src)
        if os.path.isdir(p) and p not in _sys.path:
            _sys.path.insert(0, p)


def _emit(result: dict) -> None:
    """In 1 dòng JSON cho runner (đúng prefix RESULT_JSON:)."""
    sys.stdout.write("RESULT_JSON: " + json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _diseases():
    from dl.preprocess import DISEASES  # noqa: E402  (kéo torch — lazy)

    return DISEASES


def _s3_client(endpoint: str):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "chexspark"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "chexspark123"),
    )


def _list_image_keys(endpoint: str, n: int) -> list:
    """Danh sách key ảnh PNG trên MinIO — sort để 2 engine cùng thứ tự."""
    s3 = _s3_client(endpoint)
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET_IMAGES, Prefix=PREFIX_IMAGES
    ):
        for o in page.get("Contents", []):
            if o["Key"].lower().endswith(".png"):
                keys.append(o["Key"])
    keys.sort()
    return keys[:n]


# ════════════════════════════ W1: ETL metadata ════════════════════════════
def wl_etl(args, engine: str, scale: int) -> dict:
    label_cols = [f"label_{d}" for d in _diseases()]

    if engine == "local":
        import pandas as pd
        import pyarrow.fs as pafs
        import pyarrow.parquet as pq

        ep = args.s3_endpoint.split("//", 1)[-1]
        fs = pafs.S3FileSystem(
            access_key=os.getenv("AWS_ACCESS_KEY_ID", "chexspark"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "chexspark123"),
            endpoint_override=ep,
            scheme="http",
        )
        t0 = time.perf_counter()
        df = pq.read_table("silver/cxr14_full/", filesystem=fs).to_pandas()
        if scale > 1:
            df = pd.concat([df] * scale, ignore_index=True)

        sums = df[label_cols].sum()
        res = {
            "n_rows": int(len(df)),
            "n_patients": int(df["patient_id"].nunique()),
            "disease_total": int(sums.sum()),
            "comorbid_ge2": int((df["n_diseases"] >= 2).sum()),
            "splits": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        }
    else:
        from pyspark.sql import functions as F

        spark, session_s = _get_spark("bench-etl")
        t0 = time.perf_counter()
        df = spark.read.parquet(SILVER_FULL)
        for _ in range(scale - 1):
            df = df.union(spark.read.parquet(SILVER_FULL))

        row = df.agg(
            F.count("*").alias("n_rows"),
            F.countDistinct("patient_id").alias("n_patients"),
            *[F.sum(F.col(c)).alias(c) for c in label_cols],
            F.sum((F.col("n_diseases") >= 2).cast("int")).alias("comorbid_ge2"),
        ).collect()[0]
        res = {
            "n_rows": int(row["n_rows"]),
            "n_patients": int(row["n_patients"]),
            "disease_total": int(sum(row[c] or 0 for c in label_cols)),
            "comorbid_ge2": int(row["comorbid_ge2"] or 0),
            "splits": {
                r["split"]: int(r["count"])
                for r in df.groupBy("split").count().collect()
            },
        }

    seconds = time.perf_counter() - t0
    return {
        "workload": "etl",
        "engine": engine,
        "scale": scale,
        "seconds": round(seconds, 3),
        "rows_per_s": round(res["n_rows"] / seconds, 1),
        **res,
        **({"session_s": round(_SESSION["session_s"], 3)} if engine == "spark" else {}),
    }


# ═════════════════════ W2: Preprocess ảnh (throughput) ═════════════════════
def _preproc_one(s3, key: str) -> int:
    from PIL import Image

    blob = s3.get_object(Bucket=BUCKET_IMAGES, Key=key)["Body"].read()
    img = Image.open(io.BytesIO(blob)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    img.load()
    return 1


_POOL: dict = {}


def _pool_init(endpoint: str) -> None:
    _POOL["s3"] = _s3_client(endpoint)


def _pool_worker(key: str) -> int:
    return _preproc_one(_POOL["s3"], key)


def wl_preproc(args, engine: str, n_images: int) -> dict:
    keys = _list_image_keys(args.s3_endpoint, n_images)

    if engine == "local":
        from multiprocessing import Pool, cpu_count

        procs = cpu_count()
        pool = Pool(procs, initializer=_pool_init, initargs=(args.s3_endpoint,))
        t0 = time.perf_counter()
        done = sum(pool.map(_pool_worker, keys, chunksize=16))
        pool.close()
        pool.join()
    else:
        spark, _ = _get_spark("bench-preproc")
        t0 = time.perf_counter()
        parts = max(args.parts * 2, 4)

        def _part(it):
            s3 = _s3_client(args.s3_endpoint)
            for k in it:
                yield float(_preproc_one(s3, k))

        done = int(spark.sparkContext.parallelize(keys, parts).mapPartitions(_part).sum())

    seconds = time.perf_counter() - t0
    return {
        "workload": "preproc",
        "engine": engine,
        "n_images": done,
        "seconds": round(seconds, 3),
        "imgs_per_s": round(done / seconds, 1),
        **({"session_s": round(_SESSION["session_s"], 3)} if engine == "spark" else {}),
    }


# ══════════════════ W3: Suy luận DenseNet121 (throughput) ══════════════════
def _load_production_model(mlflow_uri: str):
    """Cùng 1 model production (MLflow registry) cho cả 2 engine."""
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    import mlflow.pytorch

    return mlflow.pytorch.load_model(MLFLOW_MODEL_URI, map_location="cpu")


def _get_model(mlflow_uri: str):
    if _MODEL_CACHE.get("m") is None:
        m = _load_production_model(mlflow_uri)
        m.eval()
        _MODEL_CACHE["m"] = m
    return _MODEL_CACHE["m"]


def _input_from_bytes(blob: bytes):
    """PNG bytes → tensor (1,3,224,224) — đúng pipeline Stage-2 (mask_input=none)."""
    _add_src_path()
    import numpy as np
    from PIL import Image

    from dl.preprocess import IMG_SIZE as S, build_input

    img = Image.open(io.BytesIO(blob)).convert("L").resize((S, S))
    gray = np.asarray(img).astype(np.float32) / 255.0
    return build_input(gray, mask_input="none")


def _infer_keys(keys, s3, model, batch: int) -> int:
    import torch

    done = 0
    for i in range(0, len(keys), batch):
        chunk = keys[i: i + batch]
        xs = [
            _input_from_bytes(s3.get_object(Bucket=BUCKET_IMAGES, Key=k)["Body"].read())
            for k in chunk
        ]
        x = torch.cat(xs, dim=0)
        with torch.no_grad():
            model(x)
        done += len(chunk)
    return done


def wl_inference(args, engine: str, n_images: int) -> dict:
    import torch

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    keys = _list_image_keys(args.s3_endpoint, n_images)
    model = _get_model(args.mlflow_uri)   # nạp model ngoài vùng đo (cả 2 engine)

    if engine == "local":
        t0 = time.perf_counter()
        done = _infer_keys(keys, _s3_client(args.s3_endpoint), model, batch=args.infer_batch)
        seconds = time.perf_counter() - t0
    else:
        spark, _ = _get_spark("bench-inference")
        sc = spark.sparkContext
        parts = args.parts

        # Warm-up: nạp model trong đúng `parts` Python worker sẽ chạy task chính
        # (ngoài vùng đo — tương đương session_s bên ETL)
        sc.parallelize(range(parts), parts).map(
            lambda _: _get_model(args.mlflow_uri) is not None
        ).collect()

        def _part(it):
            torch.set_num_threads(args.torch_threads or 2)
            s3 = _s3_client(args.s3_endpoint)
            return iter([_infer_keys(list(it), s3, _get_model(args.mlflow_uri),
                                     args.infer_batch)])

        t0 = time.perf_counter()
        done = int(sc.parallelize(keys, parts).mapPartitions(_part).sum())
        seconds = time.perf_counter() - t0
    return {
        "workload": "inference",
        "engine": engine,
        "n_images": done,
        "seconds": round(seconds, 3),
        "imgs_per_s": round(done / seconds, 1),
        **({"session_s": round(_SESSION["session_s"], 3)} if engine == "spark" else {}),
    }


# ═══════════════ W4: Huấn luyện DenseNet121 (steps/giây) ═══════════════
def _train_chunk(n_steps: int, threads: int, batch: int = 16) -> dict:
    """Một process huấn luyện `n_steps` bước Adam trên dữ liệu ngẫu nhiên."""
    _add_src_path()
    import torch
    import torch.nn as nn

    torch.set_num_threads(threads)
    from dl.model import build_model

    model = build_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
        y = torch.rand(batch, 14)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return {"steps": n_steps, "task_s": time.perf_counter() - t0}


def wl_train(args, engine: str, steps: int) -> dict:
    threads = args.torch_threads or (2 if engine == "spark" else (os.cpu_count() or 4))
    batch = args.train_batch

    if engine == "local":
        t0 = time.perf_counter()
        r = _train_chunk(steps, threads, batch)
        seconds = time.perf_counter() - t0
        total_steps = r["steps"]
        task_times = [r["task_s"]]
    else:
        spark, _ = _get_spark("bench-train")
        parts = args.parts
        per = max(1, steps // parts)
        t0 = time.perf_counter()
        rs = (
            spark.sparkContext.parallelize(range(parts), parts)
            .map(lambda _: _train_chunk(per, threads, batch))
            .collect()
        )
        seconds = time.perf_counter() - t0
        total_steps = sum(r["steps"] for r in rs)
        task_times = [r["task_s"] for r in rs]
    return {
        "workload": "train",
        "engine": engine,
        "steps": total_steps,
        "batch": batch,
        "seconds": round(seconds, 3),
        "steps_per_s": round(total_steps / seconds, 3),
        "imgs_per_s": round(total_steps * batch / seconds, 1),
        "task_s_max": round(max(task_times), 3),
        **({"session_s": round(_SESSION["session_s"], 3)} if engine == "spark" else {}),
    }


# ═══════════════ W5: K-Means + PCA (sklearn vs Spark MLlib) ═══════════════
def wl_kmeans(args, engine: str, scale: int) -> dict:
    out: dict = {"workload": "kmeans", "engine": engine, "scale": scale}

    if engine == "local":
        import numpy as np
        import pyarrow.fs as pafs
        import pyarrow.parquet as pq
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        ep = args.s3_endpoint.split("//", 1)[-1]
        fs = pafs.S3FileSystem(
            access_key=os.getenv("AWS_ACCESS_KEY_ID", "chexspark"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "chexspark123"),
            endpoint_override=ep,
            scheme="http",
        )
        t0 = time.perf_counter()
        tbl = pq.read_table("gold/features/densenet121/features.parquet", filesystem=fs)
        feat_cols = [c for c in tbl.column_names if c.startswith("feat_")]
        X = tbl.select(feat_cols).to_pandas().to_numpy(dtype="float32")
        if scale > 1:
            X = np.tile(X, (scale, 1))
        Xs = StandardScaler().fit_transform(X).astype("float32")
        out["seconds_read_scale"] = round(time.perf_counter() - t0, 3)
        out["n_rows"] = int(X.shape[0])

        t1 = time.perf_counter()
        km = KMeans(n_clusters=10, random_state=42, max_iter=50, n_init=1).fit(Xs)
        out["seconds_fit"] = round(time.perf_counter() - t1, 3)
        out["inertia"] = round(float(km.inertia_), 2)

        t2 = time.perf_counter()
        pca = PCA(n_components=2).fit(Xs)
        out["seconds_pca"] = round(time.perf_counter() - t2, 3)
        out["pca_var_pc12"] = round(float(pca.explained_variance_ratio_[:2].sum()), 4)
    else:
        from pyspark.ml.clustering import KMeans as MLKMeans
        from pyspark.ml.feature import PCA as SparkPCA, StandardScaler as SparkScaler, VectorAssembler

        spark, session_s = _get_spark("bench-kmeans")
        t0 = time.perf_counter()
        df = spark.read.parquet(GOLD_FEATURES)
        for _ in range(scale - 1):
            df = df.union(spark.read.parquet(GOLD_FEATURES))
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        raw = VectorAssembler(inputCols=feat_cols, outputCol="raw_features").transform(df)
        out["n_rows"] = int(raw.count())
        scaled = SparkScaler(
            inputCol="raw_features", outputCol="features", withMean=True, withStd=True
        ).fit(raw).transform(raw).select("features")
        out["seconds_read_scale"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        km = MLKMeans(featuresCol="features", k=10, seed=42, maxIter=50).fit(scaled)
        out["seconds_fit"] = round(time.perf_counter() - t1, 3)
        out["inertia"] = round(float(km.summary.trainingCost), 2)

        t2 = time.perf_counter()
        pca = SparkPCA(k=2, inputCol="features", outputCol="pca_features").fit(scaled)
        out["seconds_pca"] = round(time.perf_counter() - t2, 3)
        out["pca_var_pc12"] = round(float(sum(pca.explainedVariance[:2])), 4)
        out["session_s"] = round(session_s, 3)

    return out


# ═══════════════════════════ session Spark ═══════════════════════════
def _get_spark(app: str):
    if _SESSION["spark"] is None:
        from common.spark_session import get_spark_session

        t0 = time.perf_counter()
        _SESSION["spark"] = get_spark_session(app)
        _SESSION["session_s"] = time.perf_counter() - t0
    return _SESSION["spark"], _SESSION["session_s"]


# ═══════════════════════════════ main ═══════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Single machine vs Spark cluster")
    p.add_argument("--engine", choices=["local", "spark"], required=True)
    p.add_argument("--workloads", default="etl,preproc,inference,train,kmeans")
    p.add_argument("--scales", default="1,5,10", help="scale × cho workload etl")
    p.add_argument("--kmeans-scales", default="1,5")
    p.add_argument("--n-images", type=int, default=2000, help="số ảnh preproc/inference")
    p.add_argument("--train-steps", type=int, default=32, help="tổng số bước train")
    p.add_argument("--train-batch", type=int, default=4,
                   help="batch train/bước (4: vừa đủ đo throughput, không gây OOM WSL)")
    p.add_argument("--infer-batch", type=int, default=16, help="batch forward inference")
    p.add_argument("--parts", type=int, default=4, help="partition cho task Spark")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch threads/process (0 = auto: local=cpu_count, spark=2)")
    p.add_argument("--s3-endpoint", default="http://localhost:9000")
    p.add_argument("--mlflow-uri", default="http://localhost:5000")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    t_start = time.perf_counter()

    for w in workloads:
        t0 = time.perf_counter()
        try:
            if w == "etl":
                for s in (int(x) for x in args.scales.split(",")):
                    _emit(wl_etl(args, args.engine, s))
            elif w == "preproc":
                _emit(wl_preproc(args, args.engine, args.n_images))
            elif w == "inference":
                _emit(wl_inference(args, args.engine, args.n_images))
            elif w == "train":
                _emit(wl_train(args, args.engine, args.train_steps))
            elif w == "kmeans":
                for s in (int(x) for x in args.kmeans_scales.split(",")):
                    _emit(wl_kmeans(args, args.engine, s))
            else:
                print(f"[skip] workload không rõ: {w}", file=sys.stderr)
        except Exception as e:  # 1 workload lỗi không giết cả run
            import traceback

            _emit({"workload": w, "engine": args.engine, "error": repr(e)})
            print(f"[error] {w}: {e!r}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        print(f"[done] {w} ({args.engine}) {time.perf_counter() - t0:.1f}s",
              file=sys.stderr, flush=True)

    _emit({
        "workload": "_summary",
        "engine": args.engine,
        "wall_s": round(time.perf_counter() - t_start, 3),
        "cpu_count": os.cpu_count(),
        "torch_threads_arg": args.torch_threads,
        "n_images": args.n_images,
        "train_steps": args.train_steps,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
