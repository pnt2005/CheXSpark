"""Runner benchmark **Single machine vs Spark cluster** — CheXSpark.

Chạy cùng `src/benchmark/spark_vs_local.py` trên 2 engine:

* **local** — `python` trên máy dev (dùng toàn bộ core, pandas/PyTorch/sklearn);
* **spark** — `spark-submit` trong container master (cluster 2 worker × 2 core,
  cùng host với máy dev — xem ghi chú "khả năng mở rộng" trong README).

Số liệu gom thành `data/outputs/spark_vs_local_benchmark.json` + bảng so sánh
in ra stdout.

Ví dụ::

    python scripts/dev/benchmark_spark_vs_local.py --quick    # smoke (~1 phút)
    python scripts/dev/benchmark_spark_vs_local.py            # full (~10 phút)
    python scripts/dev/benchmark_spark_vs_local.py --engine local   # chỉ 1 engine
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # scripts/dev/ → repo root
SCRIPT = ROOT / "src" / "benchmark" / "spark_vs_local.py"
OUT_DEFAULT = ROOT / "data" / "outputs" / "spark_vs_local_benchmark.json"

RESULT_PREFIX = "RESULT_JSON: "


def load_env() -> dict:
    """Env cho subprocess local: đọc .env (AWS key MinIO) + PYTHONIOENCODING."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    # Máy dev có AWS key THẬT trong env → phải đè bằng creds MinIO (setdefault
    # không đủ vì key đã tồn tại sẽ bị giữ lại → MinIO trả AccessDenied).
    env["AWS_ACCESS_KEY_ID"] = env.get("MINIO_ROOT_USER", "chexspark")
    env["AWS_SECRET_ACCESS_KEY"] = env.get("MINIO_ROOT_PASSWORD", "chexspark123")
    return env


def parse_results(text: str) -> list:
    out = []
    for line in text.splitlines():
        i = line.find(RESULT_PREFIX)
        if i >= 0:
            try:
                out.append(json.loads(line[i + len(RESULT_PREFIX):]))
            except json.JSONDecodeError:
                pass
    return out


def _workload_args(opts) -> list:
    return [
        "--workloads", opts.workloads,
        "--scales", opts.scales,
        "--kmeans-scales", opts.kmeans_scales,
        "--n-images", str(opts.n_images),
        "--train-steps", str(opts.train_steps),
        "--train-batch", str(opts.train_batch),
        "--infer-batch", str(opts.infer_batch),
    ]


def run_local(opts, env) -> dict:
    cmd = [
        sys.executable, str(SCRIPT), "--engine", "local",
        *_workload_args(opts),
        "--s3-endpoint", "http://localhost:9000",
        "--mlflow-uri", "http://localhost:5000",
    ]
    print(f"$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    results = parse_results(p.stdout)
    if not results:
        print("!! local không trả RESULT_JSON — stdout tail:")
        print(p.stdout[-3000:])
        print(p.stderr[-3000:], file=sys.stderr)
    if any("error" in r for r in results):
        print("!! có workload lỗi:", [r for r in results if "error" in r])
    return {"engine": "local", "wall_s": round(wall, 3), "results": results}


def run_spark(opts) -> dict:
    cmd = [
        "docker", "compose", "exec", "-T", "spark-master",
        "/opt/spark/bin/spark-submit",
        "--master", "spark://spark-master:7077",
        "--conf", "spark.ui.showConsoleProgress=false",
        "/opt/spark/user-src/benchmark/spark_vs_local.py",
        "--engine", "spark",
        *_workload_args(opts),
        "--s3-endpoint", "http://minio:9000",
        "--mlflow-uri", "http://mlflow:5000",
        "--torch-threads", "2",
    ]
    print(f"$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    merged = (p.stdout or "") + "\n" + (p.stderr or "")
    results = parse_results(merged)
    if not results:
        print("!! spark không trả RESULT_JSON — output tail:")
        print(merged[-4000:])
    if any("error" in r for r in results):
        print("!! có workload lỗi:", [r for r in results if "error" in r])
    return {"engine": "spark", "wall_s": round(wall, 3), "results": results}


def _torch_version(engine: str, env) -> str:
    try:
        if engine == "local":
            p = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"],
                               capture_output=True, text=True, encoding="utf-8")
            return p.stdout.strip()
        p = subprocess.run(["docker", "exec", "chexspark-spark-master", "python3", "-c",
                            "import torch;print(torch.__version__)"],
                           capture_output=True, text=True, encoding="utf-8")
        return p.stdout.strip()
    except Exception:
        return "?"


def _find(results: list, workload: str, scale=None) -> dict:
    for r in results:
        if r.get("workload") == workload and (scale is None or r.get("scale") == scale):
            if "error" not in r:
                return r
    return {}


def _speedup(a: float, b: float) -> tuple:
    """(engine nhanh hơn, hệ số) — số liệu cao hơn = nhanh hơn (img/s, steps/s)."""
    if a >= b:
        return "local", (a / b if b else 0)
    return "spark", (b / a if a else 0)


def print_table(payload: dict) -> None:
    loc, spr = payload["local"], payload["spark"]
    lr, sr = loc.get("results", []), spr.get("results", [])
    n = payload["meta"]["n_images"]
    steps = payload["meta"]["train_steps"]
    tb = payload["meta"].get("train_batch", 4)
    ib = payload["meta"].get("infer_batch", 16)

    print()
    print("═" * 78)
    print(" BENCHMARK: Single machine vs Spark cluster — CheXSpark")
    print(f" local: {payload['meta']['cpu_count']} vCPU, torch {payload['meta']['torch_local']}"
          f" | spark: 2 worker × 2 vCPU, torch {payload['meta']['torch_spark']}")
    print(f" ảnh: {n} (infer batch{ib}) | train: {steps} bước × batch{tb} | cluster cùng host với máy dev")
    print("═" * 78)

    # ETL
    print("\n[etl] metadata Silver 112k + aggregate (bệnh, đồng mắc, split)")
    for scale in payload["meta"]["scales"]:
        a, b = _find(lr, "etl", scale), _find(sr, "etl", scale)
        if a and b:
            ratio = a["seconds"] / b["seconds"] if b["seconds"] else 0
            if ratio < 1:
                faster, speedup = "local", (1 / ratio if ratio else 0)
            else:
                faster, speedup = "spark", ratio
            print(f"  ×{scale}: local {a['seconds']:7.2f}s ({a['rows_per_s']:>9,.0f} rows/s)"
                  f" | spark {b['seconds']:7.2f}s ({b['rows_per_s']:>9,.0f} rows/s)"
                  f" | session {b.get('session_s', 0):.1f}s"
                  f" → {faster} ×{speedup:.1f} nhanh hơn")

    # preproc
    a, b = _find(lr, "preproc"), _find(sr, "preproc")
    if a and b:
        w, s = _speedup(a["imgs_per_s"], b["imgs_per_s"])
        print(f"\n[preproc] decode + resize {n} ảnh PNG (throughput)")
        print(f"  local {a['imgs_per_s']:7.1f} img/s | spark {b['imgs_per_s']:7.1f} img/s"
              f" → {w} ×{s:.1f}")

    # inference
    a, b = _find(lr, "inference"), _find(sr, "inference")
    if a and b:
        w, s = _speedup(a["imgs_per_s"], b["imgs_per_s"])
        print(f"\n[inference] DenseNet121 production (MLflow registry) — {n} ảnh")
        print(f"  local {a['imgs_per_s']:7.1f} img/s | spark {b['imgs_per_s']:7.1f} img/s"
              f" → {w} ×{s:.1f}")

    # train
    a, b = _find(lr, "train"), _find(sr, "train")
    if a and b:
        w, s = _speedup(a["steps_per_s"], b["steps_per_s"])
        print(f"\n[train] DenseNet121 {steps} bước Adam × batch{tb} (fwd+bwd CPU)")
        print(f"  local {a['steps_per_s']:6.2f} steps/s ({a['imgs_per_s']:.0f} img/s)"
              f" | spark {b['steps_per_s']:6.2f} steps/s ({b['imgs_per_s']:.0f} img/s)"
              f" → {w} ×{s:.1f}")

    # kmeans
    print("\n[kmeans] feature 1024-d, K-Means k=10 + PCA k=2 (sklearn vs MLlib)")
    for scale in payload["meta"]["kmeans_scales"]:
        a, b = _find(lr, "kmeans", scale), _find(sr, "kmeans", scale)
        if a and b:
            print(f"  ×{scale} ({a['n_rows']:,} dòng):"
                  f" fit local {a['seconds_fit']:6.2f}s | spark {b['seconds_fit']:6.2f}s"
                  f" | inertia {a['inertia']:.3g} vs {b['inertia']:.3g}"
                  f" | PCA-2var {a['pca_var_pc12']:.3f} vs {b['pca_var_pc12']:.3f}")

    print("\n" + "─" * 78)
    print(f" wall total: local {loc['wall_s']:.0f}s | spark (spark-submit e2e) {spr['wall_s']:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Single machine vs Spark cluster")
    ap.add_argument("--engine", choices=["both", "local", "spark"], default="both")
    ap.add_argument("--workloads", default="etl,preproc,inference,train,kmeans")
    ap.add_argument("--scales", default="1,5,10")
    ap.add_argument("--kmeans-scales", default="1,5")
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--train-steps", type=int, default=32)
    ap.add_argument("--train-batch", type=int, default=4,
                    help="batch/bước train (giữ nhỏ để không OOM WSL khi 4 worker Spark cùng chạy)")
    ap.add_argument("--infer-batch", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="smoke: ít ảnh/bước/scale")
    ap.add_argument("--show", action="store_true",
                    help="chỉ in bảng từ JSON đã có (không chạy lại)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    opts = ap.parse_args()

    if opts.show:
        payload = json.loads(Path(opts.out).read_text(encoding="utf-8"))
        if "local" in payload and "spark" in payload:
            print_table(payload)
        else:
            print("JSON chưa đủ 2 engine:", list(payload.keys()))
        return 0

    if opts.quick:
        opts.scales = "1"
        opts.kmeans_scales = "1"
        opts.n_images = 128
        opts.train_steps = 8

    env = load_env()
    payload: dict = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cpu_count": os.cpu_count(),
            "spark_cluster": "2 worker × 2 vCPU (SPARK_WORKER_CORES=2), cùng host",
            "scales": [int(x) for x in opts.scales.split(",")],
            "kmeans_scales": [int(x) for x in opts.kmeans_scales.split(",")],
            "n_images": opts.n_images,
            "train_steps": opts.train_steps,
            "train_batch": opts.train_batch,
            "infer_batch": opts.infer_batch,
            "workloads": opts.workloads,
            "quick": opts.quick,
        },
    }

    if opts.engine in ("both", "local"):
        print("── run LOCAL ──")
        payload["local"] = run_local(opts, env)
    if opts.engine in ("both", "spark"):
        print("── run SPARK ──")
        payload["spark"] = run_spark(opts)

    payload["meta"]["torch_local"] = _torch_version("local", env)
    payload["meta"]["torch_spark"] = _torch_version("spark", env) if "spark" in payload else "-"

    out = Path(opts.out)
    if opts.engine != "both" and out.exists():
        # Chạy partial (1 engine / vài workload) → gộp theo từng result
        # (workload, scale): kết quả mới đè cũ, không mất số liệu đã có.
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
            for k in ("local", "spark"):
                if k in payload and k in old:
                    merged = {}
                    for r in old[k]["results"] + payload[k]["results"]:
                        merged[(r.get("workload"), r.get("scale"))] = r  # mới đè cũ
                    payload[k]["results"] = list(merged.values())
                    payload[k]["wall_s"] = round(
                        old[k].get("wall_s", 0) + payload[k].get("wall_s", 0), 3)
                    payload[k]["runs"] = old[k].get("runs", 1) + 1
                elif k not in payload and k in old:
                    payload[k] = old[k]
            if "local" in payload and "spark" in payload and "meta" in old:
                old["meta"]["merged_at"] = payload["meta"]["timestamp"]
                payload["meta"] = old["meta"]
        except Exception as e:  # noqa: BLE001
            print(f"(không gộp được JSON cũ: {e})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out}")

    if "local" in payload and "spark" in payload:
        print_table(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
