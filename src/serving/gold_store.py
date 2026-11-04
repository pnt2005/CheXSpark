"""Đọc Gold layer (Parquet trên MinIO) cho endpoint analytics của FastAPI.

Gold là output của Spark (`src/gold/*`) ghi `s3://gold/<prefix>` dạng Parquet.
Serving **không** chạy Spark chỉ để đọc vài bảng thống kê (14 hàng disease stats,
14 node / ~90 cạnh graph) — đọc trực tiếp qua boto3 + pyarrow, cache theo TTL.

Prefix hay dùng::

    stats/disease       bệnh × (cases, prevalence, male/female, pa/ap)
    stats/comorbidity   số bệnh/ảnh (0..14) × số ảnh
    stats/top_pairs     top 20 cặp đồng xuất hiện
    graph/nodes         degree, centrality, community, component từng bệnh
    graph/edges         src, dst, weight (số ca), lift, jaccard
    graph/summary       n_nodes, n_edges, modularity, top_central, ...

Cache trong process (TTL `GOLD_CACHE_TTL`, mặc định 60s) để refresh dashboard
mà không quét bucket mỗi request.
"""
from __future__ import annotations

import io
import os
import threading
import time

import pyarrow as pa
import pyarrow.parquet as pq

# Endpoint/credential MinIO — giống x-minio-creds trong docker-compose.
#
# CHỈ đọc credential MinIO chuyên biệt (GOLD_*/MINIO_*), không đọc `AWS_*`:
# trên máy dev `AWS_ACCESS_KEY_ID` có thể trỏ tài khoản AWS thật (dự án khác) →
# boto3 gửi key đó lên MinIO và nhận InvalidAccessKeyId.
S3_ENDPOINT = os.getenv("GOLD_S3_ENDPOINT") or "http://minio:9000"
BUCKET = os.getenv("GOLD_S3_BUCKET", "gold")
TTL = float(os.getenv("GOLD_CACHE_TTL", "60"))
_ACCESS_KEY = os.getenv("GOLD_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER") or "chexspark"
_SECRET_KEY = (
    os.getenv("GOLD_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD") or "chexspark123"
)

_lock = threading.Lock()
_cache: dict[str, tuple[float, list[dict]]] = {}


def _client():
    import boto3  # import lười: host dev không cài boto3 vẫn chạy được predict

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=_ACCESS_KEY,
        aws_secret_access_key=_SECRET_KEY,
    )


class GoldUnavailable(RuntimeError):
    """MinIO chưa có bucket gold / chưa chạy job Gold (`chexspark_gold_analytics`)."""


def read_gold(prefix: str, *, ttl: float | None = None, refresh: bool = False) -> list[dict]:
    """Đọc toàn bộ Parquet dưới `s3://gold/<prefix>/` → list[dict] (row-based).

    Raises:
        GoldUnavailable: bucket rỗng/không truy cập được — endpoint trả 503.
    """
    ttl = TTL if ttl is None else ttl
    key = prefix.strip("/")
    now = time.time()

    with _lock:
        hit = _cache.get(key)
        if not refresh and hit and now - hit[0] < ttl:
            return hit[1]

    try:
        s3 = _client()
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{key}/")
    except Exception as exc:
        raise GoldUnavailable(f"Không đọc được s3://{BUCKET}/{key}/: {exc}") from exc

    keys = [
        o["Key"]
        for o in resp.get("Contents", [])
        if o["Key"].endswith(".parquet") and not o["Key"].endswith("_SUCCESS")
    ]
    if not keys:
        raise GoldUnavailable(
            f"Chưa có dữ liệu Gold tại s3://{BUCKET}/{key}/ — chạy DAG "
            f"`chexspark_gold_analytics` trước."
        )

    tables = []
    try:
        for k in sorted(keys):
            body = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()
            tables.append(pq.read_table(io.BytesIO(body)))
    except Exception as exc:
        raise GoldUnavailable(f"Lỗi tải Parquet từ s3://{BUCKET}/{key}/: {exc}") from exc

    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    rows = table.to_pylist()
    with _lock:
        _cache[key] = (time.time(), rows)
    return rows


def gold_status() -> dict:
    """Báo nhanh Gold có sẵn không (dùng cho /health, không ném exception)."""
    try:
        rows = read_gold("stats/disease", ttl=30)
        return {"available": True, "disease_rows": len(rows), "endpoint": S3_ENDPOINT}
    except GoldUnavailable as exc:
        return {"available": False, "reason": str(exc), "endpoint": S3_ENDPOINT}
