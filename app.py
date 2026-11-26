"""Streamlit demo for Grinsztajn et al. (2022),
"Why do tree-based models still outperform deep learning on tabular data?"

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import benchmark as bm
from core import data as D
from core import experiments as E
from core.models import MODEL_COLORS, MODEL_NAMES
from core.transforms import rf_importance_order

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "data" / "results"
PAPER_RESULTS = RESULTS_DIR / "benchmark_total.csv"
TEAM_RESULTS = RESULTS_DIR / "team_results.csv"

st.set_page_config(page_title="Why trees beat NNs on tabular data — Demo", layout="wide")


# ====================================================================== helpers
def color(name: str) -> str:
    return MODEL_COLORS.get(name, "#7F7F7F")


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def line_with_error(summary: pd.DataFrame, x: str, x_title: str, y_title: str, category_x=False):
    """One line per model: mean over seeds, error bar = std."""
    fig = go.Figure()
    for model, g in summary.groupby("model", sort=False):
        g = g.sort_values(x)
        fig.add_trace(go.Scatter(
            x=g[x].astype(str) if category_x else g[x], y=g["mean_acc"],
            error_y=dict(type="data", array=g["std_acc"], visible=bool((g["std_acc"] > 0).any())),
            mode="lines+markers", name=model, line=dict(color=color(model.split(" (")[0]), width=2.5),
        ))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=430,
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=30))
    return fig


# ====================================================================== cached data access
@st.cache_resource(show_spinner=False)  # no copy on each rerun: some files are large
def cached_raw(name: str) -> pd.DataFrame:
    return D.load_raw(name)


@st.cache_data(show_spinner=False)
def cached_prepared(name, target, n_train, n_test, seed) -> D.Prepared:
    return D.prepare(cached_raw(name), target, n_train, n_test, seed)


@st.cache_data(show_spinner=False)
def cached_importance_order(name, target, n_train, n_test, seed) -> np.ndarray:
    data = cached_prepared(name, target, n_train, n_test, seed)
    return rf_importance_order(data.X_train, data.y_train, seed)


@st.cache_resource(show_spinner=False)
def cached_two_feature_models(name, target, n_train, n_test, seed, f1, f2):
    data = cached_prepared(name, target, n_train, n_test, seed)
    return E.fit_two_feature_models(data, f1, f2, seed)


@st.cache_data(show_spinner=False)
def cached_smoothing(cfg, cov_mults, k, robust, models, seeds):
    return E.run_smoothing(cached_prepared(*cfg), list(cov_mults), k, robust, list(models), list(seeds))


@st.cache_data(show_spinner=False)
def cached_add(cfg, fracs, models, seeds):
    return E.run_add_features(cached_prepared(*cfg), list(fracs), list(models), list(seeds))


@st.cache_data(show_spinner=False)
def cached_remove(cfg, fracs, models, seeds, fit_on_removed):
    return E.run_remove_features(cached_prepared(*cfg), list(fracs), list(models), list(seeds),
                                 fit_on_removed)


@st.cache_data(show_spinner=False)
def cached_rotation(cfg, remove_half, models, seeds):
    return E.run_rotation(cached_prepared(*cfg), remove_half, list(models), list(seeds))


@st.cache_resource(show_spinner=False)
def cached_results(path: str, mtime: float) -> pd.DataFrame:
    """Keyed by path + modification time, so an updated file is re-read."""
    return bm.read_results(path)


@st.cache_data(show_spinner=False)
def cached_curves(path: str, mtime: float, bench_name, quantile, truncate, n_shuffles, max_iter, models):
    df = cached_results(path, mtime)
    if bench_name is not None:
        df = df[df["benchmark"] == bench_name]
    sub = df[df["model_name"].isin(models)]
    norm = bm.normalize_scores(sub, quantile, truncate)
    curves, ranks, n_datasets = bm.random_search_curves(norm, n_shuffles=n_shuffles, max_iter=max_iter)
    return bm.summarize_curves(curves), ranks, n_datasets


# ====================================================================== sidebar
st.sidebar.title("Thiết lập dữ liệu")
st.sidebar.caption("Dùng cho các tab 2–5. Dữ liệu là bản đã xử lý của tác giả "
                   "(benchmark *classification, numerical features*).")

ds_name = st.sidebar.selectbox("Dataset", D.CLF_NUM_DATASETS, index=0)
if ds_name in D.LARGE_DATASETS and not D.local_path(ds_name).exists():
    st.sidebar.warning("File lớn: lần tải đầu có thể mất vài phút.")

try:
    with st.spinner(f"Đang tải {ds_name}…"):
        raw = cached_raw(ds_name)
except Exception as exc:  # network error, etc.
    st.sidebar.error(f"Không tải được dữ liệu: {exc}")
    st.sidebar.markdown(
        f"Tải tay file [{ds_name}.csv]({D.hf_url(ds_name)}) rồi đặt vào "
        f"`data/clf_num/{ds_name}.csv`, sau đó bấm **Rerun**."
    )
    st.stop()

columns = list(raw.columns)
target = st.sidebar.selectbox("Cột target", columns, index=columns.index(D.guess_target(columns)))
n_train = st.sidebar.slider("Số mẫu train", 1000, 10000, 3000, step=500,
                            help="Paper giới hạn train ở 10.000 mẫu. Ít mẫu hơn = chạy nhanh hơn.")
n_test = st.sidebar.slider("Số mẫu test", 500, 5000, 2000, step=500)
seed = int(st.sidebar.number_input("Random seed", 0, 10_000, 0))
models = st.sidebar.multiselect("Model", MODEL_NAMES, default=MODEL_NAMES)

try:
    data = cached_prepared(ds_name, target, n_train, n_test, seed)
except ValueError as exc:
    st.sidebar.error(str(exc))
    st.stop()

cfg = (ds_name, target, n_train, n_test, seed)


def skey(name: str) -> str:
    """Results are stored per data setting, so changing the sidebar hides stale results."""
    return f"{name}|{cfg}|{tuple(models)}"

st.sidebar.markdown(
    f"**{ds_name}** — {data.n_rows_available:,} dòng, {data.X_train.shape[1]} feature  \n"
    f"Train {len(data.y_train):,} · Test {len(data.y_test):,} · "
    f"tỉ lệ lớp 1 (train): {data.y_train.mean():.0%}"
)

# ====================================================================== header
st.title("Why do tree-based models still outperform deep learning on tabular data?")
st.caption("Demo cho paper của Grinsztajn, Oyallon & Varoquaux (2022). Dữ liệu và các phép "
           "biến đổi lấy từ repo của tác giả: github.com/LeoGrin/tabular-benchmark")

tab_bench, tab_db, tab_smooth, tab_feat, tab_rot = st.tabs([
    "1 · Benchmark results",
    "2 · Decision boundary",
    "3 · Finding 1: Smoothing",
    "4 · Finding 2: Uninformative features",
    "5 · Finding 3: Rotation",
])

# ====================================================================== tab 1: benchmark
with tab_bench:
    st.subheader("Figure 1–2: điểm theo số iteration của random search")
    st.markdown(
        "Mỗi đường là một model. Tại mỗi iteration *n*: lấy model tốt nhất trên **validation** "
        "trong *n* lần thử đầu và ghi điểm **test** của nó. Điểm được chuẩn hóa theo từng dataset "
        "rồi lấy trung bình. Dải màu là min–max qua các lần xáo thứ tự; đường chấm là default "
        "hyperparameters."
    )

    source = st.radio("Nguồn kết quả", ["Paper (tác giả)", "Nhóm tái hiện"], horizontal=True)
    default_path = PAPER_RESULTS if source.startswith("Paper") else TEAM_RESULTS
    path = Path(st.text_input("Đường dẫn file CSV", str(default_path.relative_to(ROOT))))
    if not path.is_absolute():
        path = ROOT / path
    uploaded = st.file_uploader("…hoặc upload file CSV", type="csv")

    if uploaded is not None:  # save the upload next to the other results, then read it like a path
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"uploaded_{uploaded.name}"
        if not path.exists() or path.stat().st_size != uploaded.size:
            path.write_bytes(uploaded.getvalue())

    results = None
    try:
        if path.exists():
            with st.spinner("Đang đọc file kết quả (file lớn, lần đầu hơi lâu)…"):
                results = cached_results(str(path), path.stat().st_mtime)
    except ValueError as exc:
        st.error(str(exc))

    if results is None:
        if source.startswith("Paper"):
            st.info(
                "Chưa có file kết quả của tác giả. Tải về bằng lệnh (chạy trong thư mục demo):\n\n"
                f"```\ncurl -L -o data/results/benchmark_total.csv {bm.PAPER_RESULTS_URL}\n```"
            )
        else:
            st.info(
                "Chưa có kết quả của nhóm. Đặt file vào `data/results/team_results.csv` với các cột: "
                "`model_name, data__keyword, mean_val_score, mean_test_score` "
                "(thêm `hp` = 'default' cho lần chạy default, `benchmark` nếu có nhiều benchmark)."
            )
    else:
        c1, c2, c3 = st.columns(3)
        if "benchmark" in results.columns:
            options = sorted(results["benchmark"].dropna().unique())
            default_b = options.index("numerical_classification_medium") \
                if "numerical_classification_medium" in options else 0
            bench_name = c1.selectbox("Benchmark", options, index=default_b)
            df_b = results[results["benchmark"] == bench_name]
        else:
            bench_name = None
            c1.write("(file không có cột `benchmark`: dùng toàn bộ)")
            df_b = results
        is_reg = "regression" in str(bench_name)
        n_shuffles = c2.slider("Số lần xáo thứ tự", 3, 30, 15)
        max_iter = c3.slider("Số iteration tối đa", 10, 400, 400, step=10)
        all_models = sorted(df_b["model_name"].dropna().unique())
        chosen = st.multiselect("Model", all_models, default=all_models)

        if chosen:
            with st.spinner("Đang tính các đường random search…"):
                summary, ranks, n_ds = cached_curves(
                    str(path), path.stat().st_mtime, bench_name,
                    0.5 if is_reg else 0.1, is_reg, n_shuffles, max_iter, tuple(chosen))

            fig = go.Figure()
            x = ranks[1:]
            for m in chosen:
                s = summary[m]
                c = color(m)
                fig.add_trace(go.Scatter(x=x, y=s["max"][1:], line=dict(width=0),
                                         showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=s["min"][1:], line=dict(width=0), fill="tonexty",
                                         fillcolor=rgba(c, 0.18), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=s["mean"][1:], name=m,
                                         line=dict(color=c, width=2.5)))
                d0 = s["mean"][0]
                if np.isfinite(d0):
                    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[d0, d0], mode="lines",
                                             line=dict(color=c, dash="dot", width=1.5),
                                             name=f"{m} (default)", showlegend=False))
            score = "R2 score" if is_reg else "accuracy"
            fig.update_xaxes(type="log", title="Number of random search iterations")
            fig.update_yaxes(title=f"Normalized test {score} of best model (on valid set)")
            fig.update_layout(height=520, legend=dict(orientation="h", y=-0.18), margin=dict(t=30),
                              title=f"{bench_name or 'results'} — {n_ds} datasets")
            st.plotly_chart(fig, use_container_width=True)

            checkpoints = [r for r in (1, 10, 100, int(ranks[-1])) if r <= ranks[-1]]
            table = pd.DataFrame({
                m: [summary[m]["mean"][0]] + [summary[m]["mean"][r] for r in checkpoints]
                for m in chosen
            }, index=["default"] + [f"n = {r}" for r in checkpoints]).T
            st.markdown("**Điểm chuẩn hóa trung bình tại một số mốc**")
            st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

# ====================================================================== tab 2: decision boundary
with tab_db:
    st.subheader("Appendix Figure 20: RandomForest học được pattern gồ ghề, MLP thì không")
    imp_order = cached_importance_order(*cfg)
    top2 = list(imp_order[-2:][::-1])
    names = data.feature_names
    c1, c2 = st.columns(2)
    f1 = c1.selectbox("Feature trục X", range(len(names)), index=int(top2[0]),
                      format_func=lambda i: names[i])
    f2 = c2.selectbox("Feature trục Y", range(len(names)), index=int(top2[1]),
                      format_func=lambda i: names[i])
    st.caption("Mặc định là 2 feature quan trọng nhất theo Random Forest. Feature đã được "
               "Gaussianize nên trục tính theo đơn vị độ lệch chuẩn.")
    c3, c4 = st.columns(2)
    xlim = c3.slider("Khung nhìn trục X", -4.0, 4.0, (-2.5, 2.5), step=0.1)
    ylim = c4.slider("Khung nhìn trục Y", -4.0, 4.0, (-2.5, 2.5), step=0.1)

    if f1 == f2:
        st.warning("Chọn hai feature khác nhau.")
    else:
        with st.spinner("Đang train RandomForest và MLP trên 2 feature…"):
            fitted = cached_two_feature_models(*cfg, f1, f2)

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"{m} — train {fitted[m]['train_acc']:.1%} · test {fitted[m]['test_acc']:.1%}"
            for m in ("RandomForest", "MLP")])
        X2 = data.X_train[:, [f1, f2]]
        inside = ((X2[:, 0] >= xlim[0]) & (X2[:, 0] <= xlim[1]) &
                  (X2[:, 1] >= ylim[0]) & (X2[:, 1] <= ylim[1]))
        pts = np.where(inside)[0][:1500]
        for col, m in enumerate(("RandomForest", "MLP"), start=1):
            gx, gy, z = E.decision_grid(fitted[m]["model"], xlim, ylim)
            fig.add_trace(go.Heatmap(x=gx, y=gy, z=z, colorscale="RdBu", zmin=0, zmax=1,
                                     showscale=(col == 2), opacity=0.85,
                                     colorbar=dict(title="P(lớp 1)")), row=1, col=col)
            for cls, mk in ((0, "#B2182B"), (1, "#2166AC")):
                sel = pts[data.y_train[pts] == cls]
                fig.add_trace(go.Scatter(x=X2[sel, 0], y=X2[sel, 1], mode="markers",
                                         marker=dict(size=4, color=mk, line=dict(width=0.3, color="white")),
                                         name=f"lớp {data.class_names[cls]}", showlegend=(col == 1)),
                              row=1, col=col)
            fig.update_xaxes(title_text=names[f1], range=list(xlim), row=1, col=col)
            fig.update_yaxes(title_text=names[f2], range=list(ylim), row=1, col=col)
        fig.update_layout(height=520, margin=dict(t=50), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Đọc hình:** RandomForest tạo các vùng hình chữ nhật, có bậc (*piece-wise constant*), "
            "nên bám được các chỗ gồ ghề của dữ liệu. MLP cho biên quyết định trơn (*spectral bias*), "
            "nên bỏ qua các pattern nhỏ. Paper báo trên electricity: RandomForest 85%, MLP 80% test accuracy."
        )

# ====================================================================== tab 3: smoothing
with tab_smooth:
    st.subheader("Finding 1 — NNs are biased to overly smooth solutions (Figure 3)")
    st.markdown(
        "Làm trơn **nhãn của train set** bằng Gaussian kernel smoother, test set giữ nguyên. "
        "Nếu cây mất điểm nhiều còn NN gần như không đổi, nghĩa là lợi thế của cây nằm ở phần "
        "gồ ghề (*irregular patterns*) của target function — phần mà NN vốn không học được."
    )
    c1, c2, c3 = st.columns(3)
    cov_mults = c1.multiselect("(Squared) lengthscale", [0.0, 0.01, 0.05, 0.1, 0.25, 0.5],
                               default=[0.0, 0.05, 0.1, 0.25])
    k = c2.slider("Số feature quan trọng nhất giữ lại", 2, 10, 5,
                  help="Paper dùng 5: kernel smoothing kém hiệu quả khi nhiều chiều.")
    n_seeds = c3.slider("Số seed (lặp lại)", 1, 5, 1, key="seeds_smooth")
    robust = st.checkbox("Ước lượng covariance bằng MinCovDet (như mô tả trong paper)", value=False,
                         help="Code gốc mặc định dùng covariance thường; paper mô tả MinCovDet.")

    if st.button("Chạy thí nghiệm smoothing", type="primary") and models and cov_mults:
        with st.spinner("Đang làm trơn nhãn và train lại các model…"):
            st.session_state[skey("smooth")] = cached_smoothing(
                cfg, tuple(sorted(cov_mults)), k, robust, tuple(models), tuple(range(seed, seed + n_seeds)))
    if skey("smooth") in st.session_state:
        res, changes = st.session_state[skey("smooth")]
        summ = E.summarize(res, ["model", "lengthscale2"])
        st.plotly_chart(line_with_error(summ, "lengthscale2", "(Squared) lengthscale của kernel",
                                        "Test accuracy", category_x=True), use_container_width=True)
        ch = changes.groupby("lengthscale2", as_index=False)["labels_changed"].mean()
        st.caption("Tỉ lệ nhãn train bị đổi sau khi làm trơn: " +
                   " · ".join(f"{r.lengthscale2:g} → {r.labels_changed:.1%}" for r in ch.itertuples()))
        st.dataframe(summ.pivot(index="model", columns="lengthscale2", values="mean_acc")
                     .style.format("{:.3f}"), use_container_width=True)

# ====================================================================== tab 4: uninformative features
with tab_feat:
    st.subheader("Finding 2 — Uninformative features affect more MLP-like NNs (Figures 4, 5)")
    mode = st.radio("Thí nghiệm", ["Thêm feature nhiễu (Figure 5b)",
                                   "Bỏ feature ít quan trọng (Figures 4, 5a)"], horizontal=True)
    n_seeds_f = st.slider("Số seed (lặp lại)", 1, 5, 1, key="seeds_feat")

    if mode.startswith("Thêm"):
        st.markdown("Thêm các cột Gaussian **không liên quan đến target**, số cột = x% số feature gốc. "
                    "Kỳ vọng: MLP tụt mạnh, cây tụt nhẹ.")
        fracs = st.multiselect("Tỉ lệ feature nhiễu thêm vào", [0.0, 0.25, 0.5, 0.75, 1.0, 2.0],
                               default=[0.0, 0.25, 0.5, 0.75, 1.0])
        if st.button("Chạy: thêm feature", type="primary") and models and fracs:
            with st.spinner("Đang train…"):
                st.session_state[skey("add")] = cached_add(cfg, tuple(sorted(fracs)), tuple(models),
                                                     tuple(range(seed, seed + n_seeds_f)))
        if skey("add") in st.session_state:
            summ = E.summarize(st.session_state[skey("add")], ["model", "fraction"])
            summ["fraction"] = summ["fraction"] * 100
            st.plotly_chart(line_with_error(summ, "fraction", "% feature nhiễu thêm vào",
                                            "Test accuracy"), use_container_width=True)
    else:
        st.markdown("Bỏ dần các feature ít quan trọng nhất (theo Random Forest). Thêm một GBT train "
                    "**chỉ trên các feature đã bỏ**: nếu điểm của nó thấp, các feature đó thật sự vô ích.")
        fracs = st.multiselect("Tỉ lệ feature bỏ đi", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                               default=[0.0, 0.2, 0.4, 0.6, 0.8])
        on_removed = st.checkbox("Thêm GBT train trên feature đã bỏ (Figure 4)", value=True)
        if st.button("Chạy: bỏ feature", type="primary") and models and fracs:
            with st.spinner("Đang train…"):
                st.session_state[skey("remove")] = cached_remove(cfg, tuple(sorted(fracs)), tuple(models),
                                                           tuple(range(seed, seed + n_seeds_f)), on_removed)
        if skey("remove") in st.session_state:
            df = st.session_state[skey("remove")].copy()
            df.loc[df["trained_on"] == "removed", "model"] = "GradientBoostingTree (chỉ feature đã bỏ)"
            summ = E.summarize(df, ["model", "fraction"])
            summ["fraction"] = summ["fraction"] * 100
            fig = line_with_error(summ, "fraction", "% feature bỏ đi (ít quan trọng trước)",
                                  "Test accuracy")
            for tr in fig.data:
                if "chỉ feature đã bỏ" in tr.name:
                    tr.line.dash = "dash"
            st.plotly_chart(fig, use_container_width=True)

# ====================================================================== tab 5: rotation
with tab_rot:
    st.subheader("Finding 3 — Data are non invariant by rotation, so should be learning procedures (Figure 6)")
    st.markdown(
        "Nhân cả train và test với **cùng một ma trận xoay ngẫu nhiên**. Thông tin không mất, nhưng "
        "mỗi cột mới là hỗn hợp của mọi cột cũ. Kỳ vọng: MLP gần như không đổi (rotation invariant), "
        "cây tụt mạnh, và thứ hạng có thể đảo ngược."
    )
    c1, c2 = st.columns(2)
    remove_half = c1.checkbox("Bỏ 50% feature ít quan trọng trước khi xoay (Figure 6b)", value=False)
    n_seeds_r = c2.slider("Số lần xoay (seed)", 1, 5, 3, key="seeds_rot")
    if st.button("Chạy thí nghiệm rotation", type="primary") and models:
        with st.spinner("Đang xoay dữ liệu và train…"):
            st.session_state[skey("rot")] = cached_rotation(cfg, remove_half, tuple(models),
                                                      tuple(range(seed, seed + n_seeds_r)))
    if skey("rot") in st.session_state:
        res = st.session_state[skey("rot")]
        summ = E.summarize(res, ["model", "condition"])
        fig = go.Figure()
        for m, g in summ.groupby("model", sort=False):
            g = g.set_index("condition").loc[["Không xoay", "Xoay ngẫu nhiên"]]
            fig.add_trace(go.Bar(x=g.index, y=g["mean_acc"], name=m, marker_color=color(m),
                                 error_y=dict(type="data", array=g["std_acc"])))
        fig.update_layout(barmode="group", yaxis_title="Test accuracy", height=430,
                          legend=dict(orientation="h", y=-0.15), margin=dict(t=30))
        lo = max(0.0, summ["mean_acc"].min() - 0.08)
        fig.update_yaxes(range=[lo, min(1.0, summ["mean_acc"].max() + 0.04)])
        st.plotly_chart(fig, use_container_width=True)

        pivot = summ.pivot(index="model", columns="condition", values="mean_acc")
        pivot["Thay đổi"] = pivot["Xoay ngẫu nhiên"] - pivot["Không xoay"]
        st.dataframe(pivot.style.format("{:+.3f}", subset=["Thay đổi"])
                     .format("{:.3f}", subset=["Không xoay", "Xoay ngẫu nhiên"]),
                     use_container_width=True)
        st.caption("MLP ở đây train bằng Adam nên chỉ bất biến *gần đúng* khi xoay: điểm có thể "
                   "lệch nhẹ (~1%) chứ không giống hệt.")
