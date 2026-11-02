"""Silver → precompute mask 224×224 (packed bits + PNG) từ `cxr14_masked`.

Vì sao cần job này
------------------
`silver/cxr14_masked` giữ mask CheXmask dạng **RLE 1024×1024**. Mỗi lần
train/eval, DataLoader phải parse chuỗi RLE (hàng nghìn cặp `start,length`),
OR 3 lớp, rồi resize — đo được **~8–12 ms/ảnh**. Trên dev 2k ảnh thì không sao,
nhưng ở 112k ảnh × 20 epoch = 2,2 triệu lần decode ≈ **6–8 giờ CPU** chỉ để
dựng lại đúng một mask không đổi. Đó là lý do phải precompute **một lần** bằng
Spark rồi train đọc lại.

Sản phẩm (1 dòng / ảnh, partition theo `split`):

    image_index      string
    split            string      (partition)
    mask_lung_bits   binary      6272 byte = np.packbits(mask phổi 224×224)
    mask_heart_bits  binary      tim (để tự OR khi cần 'lungs_heart')
    png_lung         binary      PNG xám 224×224 (xem bằng mắt / serving)
    png_lungs_heart  binary      PNG xám 224×224 (phổi + tim)
    n_fg_lung        int         số pixel phổi
    n_fg_heart       int
    img_size         int
    dice_rca_mean    double      giữ lại để lọc chất lượng mask

Packed bits nhỏ (6,1 KB/ảnh → ~0,7 GB cho 112k) và giải nén bằng
`np.unpackbits` (~0,1 ms) — nhanh hơn decode RLE ~100×.

Usage (trong container):
    spark-submit --master spark://spark-master:7077 \\
        /opt/spark/user-src/silver/precompute_masks.py

    # dev subset, giới hạn 200 dòng
    ... precompute_masks.py --limit 200
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402

SRC = os.getenv("CHEXSPARK_SILVER_MASKED", "s3a://silver/cxr14_masked")
OUT = os.getenv("CHEXSPARK_SILVER_MASKS", "s3a://silver/cxr14_masks")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "chexspark-silver-masks")

IMG_SIZE = 224
CHUNK = 128  # số ảnh gom lại trước khi yield (giữ RAM task thấp)

OUT_SCHEMA = StructType([
    StructField("image_index", StringType()),
    StructField("split", StringType()),
    StructField("mask_lung_bits", BinaryType()),
    StructField("mask_heart_bits", BinaryType()),
    StructField("png_lung", BinaryType()),
    StructField("png_lungs_heart", BinaryType()),
    StructField("n_fg_lung", IntegerType()),
    StructField("n_fg_heart", IntegerType()),
    StructField("img_size", IntegerType()),
    StructField("dice_rca_mean", DoubleType()),
])


# --------------------------------------------------------------------------- #
# Codec mask
# --------------------------------------------------------------------------- #
def decode_rle(rle, height: int, width: int) -> np.ndarray:
    """CheXmask RLE = cặp (start, length), reshape row-major (C order).

    Đã validate bằng bounding-box vs Landmarks ở `scripts/dev/export_silver_masked_to_kaggle.py`.
    """
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((height, width), dtype=np.uint8)
    vals = np.fromstring(str(rle), sep=" ", dtype=np.int64)
    flat = np.zeros(height * width, dtype=np.uint8)
    starts, lengths = vals[0::2], vals[1::2]
    for start, length in zip(starts, lengths):
        flat[start:start + length] = 1
    return flat.reshape((height, width))


def resize_mask(mask: np.ndarray, img_size: int) -> np.ndarray:
    """1024² → img_size² bằng NEAREST (y hệt `dl.preprocess.lung_mask`)."""
    if mask.shape == (img_size, img_size):
        return mask
    img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (img_size, img_size), Image.NEAREST
    )
    return (np.array(img) > 127).astype(np.uint8)


def pack_mask(mask: np.ndarray) -> bytes:
    """nhị phân (s,s) → 6272 byte (bit-packed, row-major)."""
    return np.packbits(mask.astype(np.uint8).ravel()).tobytes()


def unpack_mask(buf: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """6272 byte → (s,s) uint8 0/1 — dùng lại cho export/test."""
    return np.unpackbits(np.frombuffer(buf, dtype=np.uint8)).reshape(img_size, img_size)


def mask_to_png(mask: np.ndarray) -> bytes:
    """PNG xám 8-bit (0/255) — để xem bằng mắt & cho serving."""
    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Spark: mapInPandas giải mã song song
# --------------------------------------------------------------------------- #
def build_udf(img_size: int):
    """Trả về hàm mapInPandas: mỗi partition decode RLE → bits + PNG."""

    def _precompute(iterator):
        rows: list[dict] = []
        for pdf in iterator:
            for row in pdf.itertuples(index=False):
                h, w = int(row.mask_height), int(row.mask_width)
                lung = resize_mask(
                    decode_rle(row.mask_left_rle, h, w)
                    | decode_rle(row.mask_right_rle, h, w),
                    img_size,
                )
                heart = resize_mask(decode_rle(row.mask_heart_rle, h, w), img_size)
                both = lung | heart

                rows.append({
                    "image_index": row.image_index,
                    "split": row.split,
                    "mask_lung_bits": pack_mask(lung),
                    "mask_heart_bits": pack_mask(heart),
                    "png_lung": mask_to_png(lung),
                    "png_lungs_heart": mask_to_png(both),
                    "n_fg_lung": int(lung.sum()),
                    "n_fg_heart": int(heart.sum()),
                    "img_size": img_size,
                    "dice_rca_mean": (None if pd.isna(row.dice_rca_mean)
                                      else float(row.dice_rca_mean)),
                })
                if len(rows) >= CHUNK:
                    yield pd.DataFrame(rows, columns=[f.name for f in OUT_SCHEMA])
                    rows = []
        if rows:
            yield pd.DataFrame(rows, columns=[f.name for f in OUT_SCHEMA])

    return _precompute


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Precompute mask 224x224 (packed + PNG)")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N dòng đầu (dev)")
    ap.add_argument("--partitions", type=int, default=0,
                    help="số partition (0 = theo số core cluster)")
    args = ap.parse_args()

    spark = get_spark_session("chexspark-silver-precompute-masks")
    spark.sparkContext.setLogLevel("WARN")

    cols = ["image_index", "split", "mask_height", "mask_width", "dice_rca_mean",
            "mask_left_rle", "mask_right_rle", "mask_heart_rle"]
    df = spark.read.parquet(args.src).select(*cols)
    if args.limit:
        df = df.limit(args.limit)

    n_parts = args.partitions or int(os.getenv("SPARK_CORES_MAX", "4")) * 2
    df = df.repartition(n_parts)

    out = df.mapInPandas(build_udf(args.img_size), schema=OUT_SCHEMA)

    # Ghi theo split (train/val/test) để đọc đúng phần cần
    (out.write.mode("overwrite")
     .partitionBy("split")
     .parquet(args.out))

    # ------------------------------------------------------------------ #
    # Validate: số dòng, mask rỗng (decode lỗi), tỉ lệ phổi
    # ------------------------------------------------------------------ #
    result = spark.read.parquet(args.out)
    total_px = args.img_size * args.img_size
    stats = result.agg(
        F.count(F.lit(1)).alias("masked_rows"),
        F.sum((F.col("n_fg_lung") == 0).cast("int")).alias("empty_lung_masks"),
        F.sum((F.col("n_fg_heart") == 0).cast("int")).alias("empty_heart_masks"),
        F.avg(F.col("n_fg_lung") / total_px).alias("mean_lung_fg_frac"),
        F.avg(F.col("n_fg_heart") / total_px).alias("mean_heart_fg_frac"),
        F.avg(F.length("png_lung")).alias("mean_png_bytes"),
        F.sum((F.col("split") == "train").cast("int")).alias("train_rows"),
        F.sum((F.col("split") == "val").cast("int")).alias("val_rows"),
        F.sum((F.col("split") == "test").cast("int")).alias("test_rows"),
    ).collect()[0].asDict()

    expected = df.count()
    print("=== PRECOMPUTE MASKS STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["masked_rows"] != expected:
        raise RuntimeError(
            f"Số dòng output ({stats['masked_rows']}) != input ({expected}) — dừng!"
        )
    if stats["empty_lung_masks"]:
        raise RuntimeError(
            f"{stats['empty_lung_masks']} mask phổi rỗng — RLE decode sai, dừng!"
        )

    log_to_mlflow(stats, args)
    spark.stop()


def log_to_mlflow(stats: dict, args) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"precompute-masks-{datetime.utcnow():%Y%m%d-%H%M%S}"):
            mlflow.log_params({
                "source": args.src,
                "output": args.out,
                "img_size": args.img_size,
                "encoding": "packbits_6272B + png_gray",
            })
            mlflow.log_metrics({
                k: float(v) for k, v in stats.items() if v is not None
            })
            print(f"Logged run to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
