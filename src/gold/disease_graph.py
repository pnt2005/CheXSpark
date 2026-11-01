"""Giai đoạn DEV — Gold layer: Disease Co-occurrence Network (GraphX-style).

Xây mạng lưới liên kết 14 bệnh lồng ngực từ kết quả co-occurrence.

VẤN ĐỀ THỰC TẾ: trên full 112k ảnh, **mọi** cặp bệnh đều đồng xuất hiện ít nhất
1 lần → đồ thị đầy đủ 91/91 cạnh → degree centrality = 1.0 cho tất cả, community
detection sụp thành 1 nhóm. Co-occurrence thô KHÔNG đủ để nói lên liên kết bệnh lý.

GIẢI PHÁP: dùng **lift** (hệ số liên kết) = P(d1,d2) / (P(d1)·P(d2))
  - lift = 1  → độc lập (xuất hiện cùng đúng như ngẫu nhiên)
  - lift > 1  → xuất hiện cùng nhiều hơn ngẫu nhiên (liên kết thật)
  - lift < 1  → ít xuất hiện cùng hơn ngẫu nhiên (loại trừ nhau)
Lọc cạnh theo (cooccurrence >= MIN_COOCCURRENCE) VÀ (lift >= MIN_LIFT) rồi mới
tính degree/centrality/community trên đồ thị đã lọc.

LƯU Ý KỸ THUẬT: PySpark **không expose GraphX** (GraphX chỉ có trong Scala API).
Với đồ thị 14 nodes, ta tính graph analytics bằng Spark DataFrame + greedy
modularity (Clauset-Newman-Moore) thuần Python trên driver sau khi collect
(an toàn vì dữ liệu đồ thị cực nhỏ). Kết quả tương đương GraphX cho use-case này.

Output:
  - gold/graph/nodes/            (disease, degree, weighted_degree, sum_lift,
                                  centrality, community, component)
  - gold/graph/edges/            (src, dst, weight, lift, jaccard) — chỉ cạnh ý nghĩa
  - gold/graph/edges_all/        (91 cạnh đầy đủ + lift, phục vụ heatmap)
  - gold/graph/summary/          (n_nodes, n_edges_full, n_edges_sig, ...)
  - gold/stats/top_associations/ (top cặp theo lift, count >= MIN_COOCCURRENCE)

Env:
  MIN_COOCCURRENCE (default 50) — số ca cùng xuất hiện tối thiểu
  MIN_LIFT         (default 1.0) — lift tối thiểu (1.0 = không kém ngẫu nhiên)

Usage (trong container spark-master):
    /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 --deploy-mode client \\
        /opt/spark/user-src/gold/disease_graph.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime

from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.spark_session import get_spark_session  # noqa: E402
from ingestion.ingest_bronze_to_silver import DISEASES  # noqa: E402

GOLD_COOCC_PAIRS = "s3a://gold/cooccurrence/pairs"
GOLD_STATS_COMORBIDITY = "s3a://gold/stats/comorbidity"
GOLD_GRAPH_NODES = "s3a://gold/graph/nodes"
GOLD_GRAPH_EDGES = "s3a://gold/graph/edges"
GOLD_GRAPH_EDGES_ALL = "s3a://gold/graph/edges_all"
GOLD_GRAPH_SUMMARY = "s3a://gold/graph/summary"
GOLD_GRAPH_SWEEP = "s3a://gold/graph/threshold_sweep"
GOLD_STATS_TOP_ASSOC = "s3a://gold/stats/top_associations"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "chexspark-gold-graph")

MIN_COOCCURRENCE = int(os.getenv("MIN_COOCCURRENCE", "50"))
MIN_LIFT = float(os.getenv("MIN_LIFT", "1.0"))
# Quét ngưỡng lift để chọn đồ thị có cấu trúc cộng đồng tốt nhất.
LIFT_SWEEP = [float(x) for x in os.getenv("LIFT_SWEEP", "1.0,1.5,2.0,2.5,3.0").split(",")]
# Newman: Q >= 0.3 là có cấu trúc cộng đồng thật. Chọn ngưỡng THẤP NHẤT (đồ thị
# dày nhất) đạt Q >= MIN_MODULARITY để giữ nhiều bệnh/cạnh cho phân tích.
MIN_MODULARITY = float(os.getenv("MIN_MODULARITY", "0.3"))


def greedy_modularity_communities(nodes, edges):
    """Phát hiện cộng đồng bằng greedy modularity (Clauset-Newman-Moore).

    Bắt đầu mỗi node 1 cộng đồng, lặp: gộp cặp cộng đồng cho ΔQ > 0 lớn nhất.
    Q = Σ_c [L_c/m - (d_c/2m)²] với L_c = tổng trọng số cạnh nội bộ,
    d_c = tổng bậc (weighted degree) của các node trong cộng đồng, m = tổng
    trọng số toàn đồ thị.

    ΔQ(i,j) = w_ij/m - (d_i·d_j)/(2m²)
    """
    if not edges:
        return {n: n for n in nodes}

    weight = defaultdict(float)  # weight[(u,v)] tổng trọng số cạnh giữa u, v
    degree = {n: 0.0 for n in nodes}
    for u, v, w in edges:
        key = tuple(sorted((u, v)))
        weight[key] += w
        degree[u] += w
        degree[v] += w

    m = sum(weight.values()) / 2.0  # mỗi cạnh đếm 1 lần
    if m <= 0:
        return {n: n for n in nodes}

    comm = {n: n for n in nodes}          # node → community id
    comm_degree = dict(degree)            # community id → tổng bậc
    comm_self = {n: 0.0 for n in nodes}   # community id → tổng trọng số nội bộ
    # weight_between[(ci, cj)] = tổng trọng số cạnh nối 2 cộng đồng
    weight_between = defaultdict(float)
    for (u, v), w in weight.items():
        weight_between[tuple(sorted((comm[u], comm[v])))] += w

    while True:
        best_gain, best_pair = 0.0, None
        for (ci, cj), w_ij in weight_between.items():
            if ci == cj:
                continue
            gain = w_ij / m - (comm_degree[ci] * comm_degree[cj]) / (2.0 * m * m)
            if gain > best_gain or (
                gain == best_gain and best_pair is not None and (ci, cj) < best_pair
            ):
                best_gain, best_pair = gain, (ci, cj)

        if best_pair is None:
            break

        ci, cj = best_pair
        # Gộp cj vào ci (giữ id nhỏ hơn để tất định)
        keep, drop = (ci, cj) if ci < cj else (cj, ci)
        for n in nodes:
            if comm[n] == drop:
                comm[n] = keep
        comm_degree[keep] += comm_degree.pop(drop)
        comm_self[keep] = (
            comm_self.get(keep, 0.0)
            + comm_self.pop(drop, 0.0)
            + weight_between.get(tuple(sorted((keep, drop))), 0.0)
        )

        # Cập nhật trọng số giữa cộng đồng
        new_between = defaultdict(float)
        for (a, b), w in weight_between.items():
            if a == drop:
                a = keep
            if b == drop:
                b = keep
            new_between[tuple(sorted((a, b)))] += w
        weight_between = new_between

    return comm


def connected_components(nodes, edges):
    """Connected components bằng union-find (đồ thị nhỏ, chạy driver)."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v, _ in edges:
        union(u, v)
    return {n: find(n) for n in nodes}


def modularity(nodes, edges, comm):
    """Tính modularity Q của một phân hoạch cộng đồng (để log/report)."""
    if not edges:
        return 0.0
    m = sum(w for _, _, w in edges) / 2.0
    if m <= 0:
        return 0.0
    internal = defaultdict(float)
    deg = defaultdict(float)
    for u, v, w in edges:
        deg[u] += w
        deg[v] += w
        if comm[u] == comm[v]:
            internal[comm[u]] += w
    q = 0.0
    for c in set(comm.values()):
        d_c = sum(deg[n] for n in nodes if comm[n] == c)
        q += internal[c] / m - (d_c / (2.0 * m)) ** 2
    return q


def main() -> None:
    spark = get_spark_session("chexspark-gold-graph")
    spark.sparkContext.setLogLevel("WARN")

    # 1) Tổng số ảnh (từ comorbidity: tổng các bin = tổng ảnh)
    total = (
        spark.read.parquet(GOLD_STATS_COMORBIDITY)
        .agg(F.sum("images").alias("total"))
        .collect()[0]["total"]
    )

    # 2) Đọc cặp đồng xuất hiện + tính lift
    pairs = spark.read.parquet(GOLD_COOCC_PAIRS)
    pairs_with_lift = pairs.withColumn(
        "lift",
        (F.col("cooccurrence") * F.lit(float(total)))
        / (F.col("count_1") * F.col("count_2")),
    )
    all_rows = pairs_with_lift.select(
        "disease_1", "disease_2", "cooccurrence", "jaccard", "lift",
        "p_d2_given_d1", "p_d1_given_d2",
    ).collect()

    edges_all = [
        (r["disease_1"], r["disease_2"], int(r["cooccurrence"]),
         float(r["lift"]), float(r["jaccard"]))
        for r in all_rows
    ]
    # 3) Quét ngưỡng lift — chọn đồ thị có cấu trúc cộng đồng rõ ràng.
    #    Đồ thị đầy đủ (91 cạnh) không có cấu trúc; ngưỡng lift cao → mạng thưa,
    #    cộng đồng bệnh lý tách bạch nhưng rớt bệnh. Tiêu chí chọn: lấy ngưỡng
    #    thấp nhất (đồ thị dày nhất, giữ nhiều bệnh) mà modularity Q >= 0.3
    #    (Newman: Q >= 0.3 = có cấu trúc cộng đồng thật). Fallback: Q cao nhất.
    sweep_rows = []
    candidates = []
    for thr in LIFT_SWEEP:
        es = [e for e in edges_all if e[2] >= MIN_COOCCURRENCE and e[3] >= thr]
        g = [(u, v, lift) for u, v, _, lift, _ in es]
        cm = greedy_modularity_communities(DISEASES, g)
        q = modularity(DISEASES, g, cm)
        sweep_rows.append((thr, len(es), len(set(cm.values())), q))
        candidates.append((thr, q, es, g, cm))
        print(f"  [sweep] lift>={thr}: edges={len(es)} communities={len(set(cm.values()))} Q={q:.4f}")

    viable = [c for c in candidates if c[1] >= MIN_MODULARITY and c[2]]
    if viable:
        # Ưu tiên đồ thị dày nhất (nhiều cạnh) trong các ngưỡng đạt Q >= 0.3
        best = max(viable, key=lambda c: (len(c[2]), c[1]))
    else:
        best = max(candidates, key=lambda c: c[1])

    chosen_thr, q_value, edges_sig, sig_graph, communities = best
    print(f"  [chosen] lift>={chosen_thr}: edges={len(edges_sig)} "
          f"Q={q_value:.4f} (ngưỡng cấu trúc {MIN_MODULARITY})")
    if not edges_sig:  # fallback: nếu mọi ngưỡng đều rỗng
        edges_sig = [e for e in edges_all if e[2] >= MIN_COOCCURRENCE]
        sig_graph = [(u, v, lift) for u, v, _, lift, _ in edges_sig]
        communities = greedy_modularity_communities(DISEASES, sig_graph)
        q_value = modularity(DISEASES, sig_graph, communities)

    # 4) Node metrics trên ĐỒ THỊ ĐÃ CHỌN
    degree = {d: 0 for d in DISEASES}
    weighted = {d: 0 for d in DISEASES}
    sum_lift = {d: 0.0 for d in DISEASES}
    for u, v, w, lift, _ in edges_sig:
        degree[u] += 1
        degree[v] += 1
        weighted[u] += w
        weighted[v] += w
        sum_lift[u] += lift
        sum_lift[v] += lift

    n_nodes = len(DISEASES)
    components = connected_components(DISEASES, sig_graph)

    node_rows = [
        (
            d,
            degree[d],
            weighted[d],
            sum_lift[d],
            degree[d] / (n_nodes - 1) if n_nodes > 1 else 0.0,
            communities[d],
            components[d],
        )
        for d in DISEASES
    ]
    node_schema = (
        "disease string, degree int, weighted_degree long, sum_lift double, "
        "centrality double, community string, component string"
    )
    nodes_df = spark.createDataFrame(node_rows, node_schema)

    # 4) Edges Gold (đã lọc) + edges_all (91 cạnh, phục vụ heatmap)
    edge_rows = list(edges_sig)
    edge_schema = "src string, dst string, weight long, lift double, jaccard double"
    edges_df = spark.createDataFrame(edge_rows, edge_schema)

    all_edges_df = spark.createDataFrame(edges_all, edge_schema)

    # 5) Top associations theo lift (chỉ cặp đủ số ca — tránh lift nhiễu khi count nhỏ)
    top_assoc = sorted(
        [r for r in all_rows if r["cooccurrence"] >= MIN_COOCCURRENCE],
        key=lambda r: r["lift"], reverse=True,
    )[:20]
    top_assoc_rows = [
        (r["disease_1"], r["disease_2"], int(r["cooccurrence"]),
         float(r["lift"]), float(r["jaccard"]),
         float(r["p_d2_given_d1"]), float(r["p_d1_given_d2"]))
        for r in top_assoc
    ]
    top_schema = (
        "disease_1 string, disease_2 string, cooccurrence long, lift double, "
        "jaccard double, p_d2_given_d1 double, p_d1_given_d2 double"
    )
    top_assoc_df = spark.createDataFrame(top_assoc_rows, top_schema)

    # 7) Summary
    n_communities = len(set(communities.values()))
    n_components = len(set(components.values()))
    top_central = sorted(DISEASES, key=lambda d: (weighted[d], sum_lift[d]), reverse=True)[:5]
    top_lift_pair = f"{top_assoc[0]['disease_1']}+{top_assoc[0]['disease_2']}" if top_assoc else ""
    summary_rows = [(
        n_nodes, len(edges_all), len(edge_rows), chosen_thr, n_communities,
        n_components, q_value, ", ".join(top_central), top_lift_pair,
    )]
    summary_schema = (
        "n_nodes int, n_edges_full int, n_edges_significant int, "
        "lift_threshold double, n_communities int, n_components int, "
        "modularity double, top_central string, top_lift_pair string"
    )
    summary_df = spark.createDataFrame(summary_rows, summary_schema)

    sweep_schema = "lift_threshold double, n_edges int, n_communities int, modularity double"
    sweep_df = spark.createDataFrame(sweep_rows, sweep_schema)

    # 8) Ghi Gold
    nodes_df.write.mode("overwrite").parquet(GOLD_GRAPH_NODES)
    edges_df.write.mode("overwrite").parquet(GOLD_GRAPH_EDGES)
    all_edges_df.write.mode("overwrite").parquet(GOLD_GRAPH_EDGES_ALL)
    summary_df.write.mode("overwrite").parquet(GOLD_GRAPH_SUMMARY)
    sweep_df.write.mode("overwrite").parquet(GOLD_GRAPH_SWEEP)
    top_assoc_df.write.mode("overwrite").parquet(GOLD_STATS_TOP_ASSOC)

    # 9) In kết quả
    print(f"=== GRAPH NODES (chosen: count>={MIN_COOCCURRENCE}, lift>={chosen_thr}) ===")
    for r in sorted(node_rows, key=lambda x: x[2], reverse=True):
        print(f"  {r[0]}: degree={r[1]} weighted={r[2]} sum_lift={r[3]:.1f} "
              f"centrality={r[4]:.3f} community={r[5]}")
    print("=== COMMUNITIES (greedy modularity) ===")
    by_comm = defaultdict(list)
    for d in DISEASES:
        by_comm[communities[d]].append(d)
    for c, members in sorted(by_comm.items()):
        print(f"  {c}: {members}")
    print("=== TOP 10 ASSOCIATIONS BY LIFT ===")
    for r in top_assoc_rows[:10]:
        print(f"  {r[0]} + {r[1]}: count={r[2]} lift={r[3]:.2f} jaccard={r[4]:.3f}")
    print(f"=== SUMMARY: {n_nodes} nodes | edges {len(edges_all)} full → "
          f"{len(edge_rows)} significant | {n_communities} communities | "
          f"modularity Q={q_value:.3f} | {n_components} components ===")

    # 10) MLflow
    log_to_mlflow({
        "graph_nodes": n_nodes,
        "graph_edges_full": len(edges_all),
        "graph_edges_significant": len(edge_rows),
        "graph_communities": n_communities,
        "graph_components": n_components,
        "graph_modularity": q_value,
        "graph_lift_threshold": chosen_thr,
    })

    spark.stop()


def log_to_mlflow(metrics: dict) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"gold-graph-{datetime.utcnow():%Y%m%d-%H%M%S}"):
            mlflow.log_params({
                "layer": "gold",
                "analysis": "disease_graph",
                "algorithm": "greedy_modularity_cnm",
                "edge_weight": "lift",
                "min_cooccurrence": MIN_COOCCURRENCE,
                "lift_sweep": ",".join(str(x) for x in LIFT_SWEEP),
                "threshold_selection": "max_modularity",
                "source": GOLD_COOCC_PAIRS,
                "note": "PySpark has no GraphX API; graph analytics via DataFrame + driver CNM",
            })
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            print(f"Logged run to MLflow experiment '{MLFLOW_EXPERIMENT}'")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
