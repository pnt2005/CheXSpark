"""Streamlit — dashboard chẩn đoán CheXSpark (gọi FastAPI, không tự chạy model).

Giao diện cho bác sĩ (xem skill `chexspark-pipeline` — mục Streamlit Dashboard):

    1. Tải ảnh CXR (PNG/JPEG) → gọi `POST /api/v1/predict`
    2. Hiển thị 4 cột: ảnh gốc │ mask phổi (U-Net) │ vùng phổi │ heatmap Grad-CAM
    3. Thẻ xác suất 14 bệnh kèm ngưỡng Youden + cờ dương tính
    4. Khám phá mạng đồng xuất hiện bệnh lý (Gold layer / GraphX)

Chạy (host)::

    streamlit run src/serving/streamlit_app.py            # API_BASE=http://localhost:8000

hoặc qua Docker Compose (service `streamlit-app`, port 8501).
"""
from __future__ import annotations

import base64
import io
import math
import os
import sys
from pathlib import Path

# Cho phép `import dl.*` khi chạy trực tiếp file này (streamlit thêm thư mục chứa
# script vào sys.path, tức .../src/serving — cần .../src)
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

from dl.preprocess import DISEASES

API_BASE = (os.getenv("API_BASE", "http://localhost:8000")).rstrip("/")
API = f"{API_BASE}/api/v1"

st.set_page_config(
    page_title="CheXSpark — Chẩn đoán CXR",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Gọi API (cache để chuyển bệnh vẽ heatmap không phải inference lại)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30, show_spinner=False)
def _health() -> dict:
    try:
        return requests.get(f"{API}/health", timeout=10).json()
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}


@st.cache_data(ttl=120, show_spinner=False)
def _model_info() -> dict:
    try:
        return requests.get(f"{API}/model", timeout=30).json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@st.cache_data(ttl=600, max_entries=8, show_spinner=False)
def _predict(payload: bytes, disease: str | None, gradcam: bool) -> dict:
    """Không cache theo thời gian inference → chuyển bệnh chỉ tốn 1 request mới."""
    resp = requests.post(
        f"{API}/predict",
        files={"file": ("cxr.png", payload, "image/png")},
        params={"disease": disease, "gradcam": gradcam},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60, show_spinner=False)
def _gold(path: str) -> dict:
    resp = requests.get(f"{API}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _img(b64: str | None):
    if not b64:
        return None
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# --------------------------------------------------------------------------- #
# Sidebar — trạng thái service + model
# --------------------------------------------------------------------------- #
st.sidebar.title("CheXSpark 🩻")
st.sidebar.caption("Phát hiện đa bệnh lý lồng ngực từ X-quang ngực thẳng")

health = _health()
if health.get("model_loaded"):
    st.sidebar.success(f"Model sẵn sàng · device `{health.get('device')}`")
else:
    st.sidebar.error(
        f"API chưa sẵn sàng — `{API_BASE}`\n\n{health.get('detail') or health.get('status')}"
    )
gold = health.get("gold") or {}
if gold.get("available"):
    st.sidebar.caption(f"Gold layer ✓ ({gold.get('disease_rows')} dòng thống kê)")
else:
    st.sidebar.caption("Gold layer chưa có dữ liệu (chạy `chexspark_gold_analytics`)")

info = _model_info()
if "error" not in info:
    with st.sidebar.expander("Model đang serve"):
        s2 = info.get("stage2", {})
        st.write(f"- **Stage-2**: `{s2.get('source')}` — `{s2.get('ckpt') or s2.get('uri', '')}`")
        s1 = info.get("stage1", {})
        st.write(f"- **Stage-1**: `{s1.get('source')}`")
        st.write(f"- **Input**: `mask_input={info.get('mask_input')}`, heart proxy `{info.get('heart_mode')}`")
        cal = info.get("calibration", {})
        st.write(f"- **Calibration**: {cal.get('method') or 'không có'}")
        if cal.get("test_mauroc"):
            st.write(f"- **test mAUROC**: {cal['test_mauroc']:.4f}")

use_default_thr = st.sidebar.checkbox(
    "Dùng ngưỡng Youden của model", value=True,
    help="Ngưỡng per-label fit trên val lúc train. Tắt để tự chọn ngưỡng chung.",
)
custom_thr = None
if not use_default_thr:
    custom_thr = st.sidebar.slider("Ngưỡng chung", 0.05, 0.95, 0.50, 0.01)

st.sidebar.divider()
st.sidebar.caption(f"API: `{API_BASE}`")


# --------------------------------------------------------------------------- #
# Tab 1 — Chẩn đoán
# --------------------------------------------------------------------------- #
tab_diag, tab_stats, tab_graph = st.tabs(
    ["🔬 Chẩn đoán", "📊 Phân bố bệnh", "🕸 Mạng đồng xuất hiện"]
)

with tab_diag:
    uploaded = st.file_uploader(
        "Tải ảnh X-quang ngực thẳng (PNG/JPEG)", type=["png", "jpg", "jpeg"], key="cxr"
    )

    if uploaded is None:
        st.info(
            "Tải một ảnh CXR để chạy pipeline 3 giai đoạn: "
            "**U-Net phân đoạn phổi → DenseNet121 phân loại 14 bệnh → Grad-CAM**."
        )
    else:
        payload = uploaded.getvalue()
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**Ảnh gốc (224×224)**")
            st.image(payload, use_container_width=True)

        try:
            result = _predict(payload, None, True)
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text[:300]
            st.error(f"Predict lỗi {exc.response.status_code}: {detail}")
            st.stop()
        except Exception as exc:
            st.error(f"Không gọi được API `{API_BASE}`: {type(exc).__name__}: {exc}")
            st.stop()

        with col_right:
            st.markdown("**Heatmap Grad-CAM**")
            cam_name = st.selectbox(
                "Bệnh vẽ heatmap", DISEASES,
                index=DISEASES.index(result["gradcam_disease"]),
                key="cam_sel",
            )
            if cam_name != result["gradcam_disease"]:
                try:
                    result = _predict(payload, cam_name, True)
                except Exception as exc:
                    st.warning(f"Không vẽ được heatmap {cam_name}: {exc}")
            st.image(_img(result["images"]["heatmap"]), use_container_width=True)
            st.caption(f"Vùng nghi ngờ cho **{result['gradcam_disease']}** (chỉ giữ trong phổi)")

        # 4 ảnh đối chiếu
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(_img(result["images"]["original"]), caption="Ảnh xám", use_container_width=True)
        c2.image(_img(result["images"]["mask"]), caption="Mask phổi (U-Net)", use_container_width=True)
        c3.image(_img(result["images"]["masked"]), caption="Vùng phổi", use_container_width=True)
        c4.image(_img(result["images"]["heatmap"]), caption="Grad-CAM overlay", use_container_width=True)

        # Bảng xác suất
        df = pd.DataFrame(result["findings"])
        if custom_thr is not None:
            df["threshold"] = custom_thr
            df["is_positive"] = df["probability"] >= custom_thr
        df = df.sort_values("probability", ascending=False).reset_index(drop=True)
        df.insert(0, "Kết quả", df["is_positive"].map({True: "🔴 Dương", False: "—"}))

        n_pos = int(df["is_positive"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bệnh nghi ngờ", n_pos)
        m2.metric("Xác suất cao nhất", f"{df['probability'].max():.3f}", result["top_disease"])
        m3.metric("Độ phủ phổi", f"{result['mask_coverage']:.1%}")
        m4.metric("Thời gian", f"{result['timings_ms']['total']:.0f} ms")

        st.markdown("**14 bệnh lồng ngực**")
        st.dataframe(
            df[["Kết quả", "disease_name", "probability", "threshold"]],
            column_config={
                "probability": st.column_config.ProgressColumn(
                    "Xác suất", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "threshold": "Ngưỡng",
                "disease_name": "Bệnh",
            },
            hide_index=True,
            use_container_width=True,
            height=430,
        )

        with st.expander("Chi tiết thời gian & model"):
            st.write(result["timings_ms"])
            st.json(result["model"])


# --------------------------------------------------------------------------- #
# Tab 2 — Phân bố bệnh (Gold layer)
# --------------------------------------------------------------------------- #
with tab_stats:
    st.markdown("Thống kê **112.120 ảnh NIH ChestX-ray14** — output của Spark Gold layer")
    try:
        data = _gold("/analytics/disease-distribution")
        dis = pd.DataFrame(data["diseases"]).sort_values("cases", ascending=False)
        com = pd.DataFrame(data["comorbidity"])

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                go.Figure(go.Bar(x=dis["cases"], y=dis["disease"], orientation="h",
                                 marker_color="#4C78A8")).update_layout(
                    height=420, yaxis={"autorange": "reversed"},
                    title="Số ca từng bệnh", margin=dict(l=10, r=10, t=40, b=10)),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                go.Figure(go.Bar(x=com["n_diseases"], y=com["images"],
                                 marker_color="#F58518")).update_layout(
                    height=420, title="Số bệnh trên mỗi ảnh (0 = No Finding)",
                    xaxis_title="Số bệnh", yaxis_title="Số ảnh",
                    margin=dict(l=10, r=10, t=40, b=10)),
                use_container_width=True,
            )
        st.dataframe(
            dis[["disease", "cases", "prevalence", "male", "female", "pa", "ap"]].rename(columns={
                "disease": "Bệnh", "cases": "Số ca", "prevalence": "Tỷ lệ",
                "male": "Nam", "female": "Nữ", "pa": "PA", "ap": "AP",
            }),
            hide_index=True, use_container_width=True,
        )
    except requests.HTTPError as exc:
        try:
            st.warning(exc.response.json().get("detail", str(exc)))
        except Exception:
            st.warning(str(exc))
    except Exception as exc:
        st.warning(f"Không đọc được analytics: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Tab 3 — Mạng đồng xuất hiện (GraphX)
# --------------------------------------------------------------------------- #
with tab_graph:
    st.markdown(
        "Nodes = 14 bệnh · Cạnh = tần suất đồng xuất hiện (đã lọc theo lift) · "
        "**output của Spark GraphX**"
    )
    try:
        g = _gold("/graph/co-occurrence")
        nodes, edges = pd.DataFrame(g["nodes"]), pd.DataFrame(g["edges"])
        summary = (g.get("summary") or [{}])[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cạnh (đã lọc)", summary.get("n_edges_significant", len(edges)))
        c2.metric("Cộng đồng", summary.get("n_communities", "—"))
        c3.metric("Thành phần liên thông", summary.get("n_components", "—"))
        c4.metric("Modularity", f"{summary.get('modularity', 0):.3f}")
        if summary.get("top_central"):
            st.caption(f"Bệnh trung tâm nhất: **{summary['top_central']}** · Cặp lift cao: **{summary.get('top_lift_pair', '—')}**")

        # Layout vòng tròn — node to theo degree, màu theo community
        n = len(nodes)
        nodes = nodes.copy()
        nodes["angle"] = [2 * math.pi * i / n for i in range(n)]
        nodes["x"] = [math.cos(a) for a in nodes["angle"]]
        nodes["y"] = [math.sin(a) for a in nodes["angle"]]
        pos = {r.disease: (r.x, r.y) for r in nodes.itertuples()}

        # Cạnh: mỗi đoạn = (src, dst, None) để plotly vẽ đường đứt đoạn riêng lẻ
        ex: list = []
        ey: list = []
        for e in edges.to_dict("records"):
            x1, y1 = pos[e["src"]]
            x2, y2 = pos[e["dst"]]
            ex += [x1, x2, None]
            ey += [y1, y2, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=1, color="rgba(150,150,150,0.6)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=nodes["x"], y=nodes["y"], mode="markers+text",
            text=nodes["disease"], textposition="middle center",
            # list màu (không phải tên "Set2" — plotly chỉ nhận continuous colorscale
            # cho marker.colorscale) → mỗi cộng đồng một màu
            marker=dict(size=[14 + 4 * d for d in nodes["degree"]],
                        color=nodes["community"].astype("category").cat.codes,
                        colorscale=list(pc.qualitative.Set2),
                        line=dict(width=1, color="white")),
            customdata=nodes[["degree", "centrality", "community"]].values,
            hovertemplate="%{text}<br>degree=%{customdata[0]}<br>centrality=%{customdata[1]:.2f}"
                          "<br>community=%{customdata[2]}<extra></extra>",
        ))
        fig.update_layout(
            height=560, showlegend=False, hovermode="closest",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Cạnh mạnh nhất**")
            st.dataframe(
                edges.sort_values("weight", ascending=False).head(15).rename(columns={
                    "src": "Bệnh 1", "dst": "Bệnh 2", "weight": "Số ca chung",
                    "lift": "Lift", "jaccard": "Jaccard",
                }),
                hide_index=True, use_container_width=True,
            )
        with right:
            st.markdown("**Degree centrality**")
            st.dataframe(
                nodes[["disease", "degree", "centrality", "community"]]
                .sort_values("degree", ascending=False).rename(columns={
                    "disease": "Bệnh", "degree": "Degree",
                    "centrality": "Centrality", "community": "Cộng đồng",
                }),
                hide_index=True, use_container_width=True,
            )
    except requests.HTTPError as exc:
        try:
            st.warning(exc.response.json().get("detail", str(exc)))
        except Exception:
            st.warning(str(exc))
    except Exception as exc:
        st.warning(f"Không đọc được graph: {type(exc).__name__}: {exc}")
