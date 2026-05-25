"""
Streamlit dashboard for the stock-recommendation system.

Reads the CSV artifacts produced by `python main.py`:
  - outputs/metrics.csv               aggregate (mean/std/min/max) per model
  - outputs/walk_forward_results.csv  one row per (model, window)
  - outputs/predictions.csv           pooled OOS predictions
  - outputs/losses.csv                train + val loss per epoch
  - outputs/per_stock/<model>.csv

Run with:
    streamlit run dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime import exists as _streamlit_runtime_exists

if not _streamlit_runtime_exists():
    sys.stderr.write(
        "\nThis file is a Streamlit app — it must be launched with the Streamlit CLI:\n"
        "    streamlit run dashboard.py\n\n"
        "Running it with plain `python dashboard.py` (or the IDE Run button) only "
        "prints 'missing ScriptRunContext' warnings and renders nothing.\n"
    )
    sys.exit(1)

OUTPUTS_DIR = Path("./outputs")
METRICS_CSV = OUTPUTS_DIR / "metrics.csv"
WALK_FORWARD_CSV = OUTPUTS_DIR / "walk_forward_results.csv"
PREDICTIONS_CSV = OUTPUTS_DIR / "predictions.csv"
LOSSES_CSV = OUTPUTS_DIR / "losses.csv"
PER_STOCK_DIR = OUTPUTS_DIR / "per_stock"


st.set_page_config(page_title="Stock LSTM Dashboard", layout="wide")
st.title("Stock LSTM — Analyst Dashboard")


@st.cache_data(show_spinner=False)
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _missing_outputs_banner() -> bool:
    missing = [
        p
        for p in (METRICS_CSV, WALK_FORWARD_CSV, PREDICTIONS_CSV, LOSSES_CSV)
        if not p.exists()
    ]
    if missing:
        st.info(
            "No analysis artifacts found yet. Run `python main.py` first to generate "
            f"metrics. Missing: {', '.join(str(m) for m in missing)}"
        )
        return True
    return False


if _missing_outputs_banner():
    st.stop()


metrics_df = _load_csv(METRICS_CSV).set_index("model")
wf_df = _load_csv(WALK_FORWARD_CSV)
wf_df["window_start"] = pd.to_datetime(wf_df["window_start"])
predictions_df = _load_csv(PREDICTIONS_CSV)
predictions_df["date"] = pd.to_datetime(predictions_df["date"])
losses_df = _load_csv(LOSSES_CSV)

# ---------- KPI cards ----------
da_col = "DirectionalAccuracy_mean"
rmse_col = "RMSE_mean"

best_da_model = metrics_df[da_col].idxmax()
best_rmse_model = metrics_df[rmse_col].idxmin()
naive_da = metrics_df.loc["Naive", da_col] if "Naive" in metrics_df.index else float("nan")
lift_vs_naive = metrics_df.loc[best_da_model, da_col] - naive_da if pd.notna(naive_da) else float("nan")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(
    "Best Success Rate",
    f"{metrics_df.loc[best_da_model, da_col]:.1f}%",
    f"{best_da_model}"
    + (f" (+{lift_vs_naive:.1f}pp vs Naive)" if pd.notna(lift_vs_naive) else ""),
)
kpi2.metric(
    "Best RMSE ($)",
    f"{metrics_df.loc[best_rmse_model, rmse_col]:.2f}",
    f"by {best_rmse_model}",
)
kpi3.metric(
    "Naive baseline",
    f"{naive_da:.1f}%" if pd.notna(naive_da) else "n/a",
    "must beat this",
)

if pd.notna(lift_vs_naive) and lift_vs_naive < 0:
    st.error(
        f"⚠️ Best model ({best_da_model}, {metrics_df.loc[best_da_model, da_col]:.1f}%) "
        f"is BELOW the Naive baseline ({naive_da:.1f}%). Models are likely overfitting noise."
    )
elif pd.notna(lift_vs_naive) and lift_vs_naive < 1.5:
    st.warning(
        f"Best LSTM beats Naive by only {lift_vs_naive:.1f}pp — within walk-forward noise."
    )
elif pd.notna(lift_vs_naive):
    st.success(
        f"{best_da_model} beats the Naive baseline by {lift_vs_naive:.1f}pp."
    )

st.divider()

tab_models, tab_stocks, tab_train, tab_wf = st.tabs(
    ["Model comparison", "Per-stock drill-down", "Training diagnostics", "Walk-forward stability"]
)

# ---------- Tab 1: model comparison ----------
with tab_models:
    st.subheader("Model comparison (walk-forward mean)")
    comp = metrics_df[[rmse_col, "MAPE_mean", da_col]].reset_index().rename(
        columns={"model": "Model", rmse_col: "RMSE", "MAPE_mean": "MAPE%", da_col: "DirAcc%"}
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.bar(comp, x="Model", y="RMSE", color="Model", title="RMSE ($)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(comp, x="Model", y="MAPE%", color="Model", title="MAPE (%)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.bar(
            comp, x="Model", y="DirAcc%", color="Model",
            title="Directional Accuracy (success rate, %)",
            error_y=metrics_df["DirectionalAccuracy_std"].reindex(comp["Model"]).values,
        )
        fig.add_hline(y=50, line_dash="dash", line_color="grey", annotation_text="random")
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

# ---------- Tab 2: per-stock drill-down ----------
with tab_stocks:
    st.subheader("Per-stock drill-down")
    tickers = sorted(predictions_df["ticker"].unique())
    selected = st.selectbox("Ticker", tickers)

    ticker_df = predictions_df[predictions_df["ticker"] == selected].sort_values(["model", "date"])

    fig = go.Figure()
    first_model = ticker_df["model"].iloc[0]
    actual_one = ticker_df[ticker_df["model"] == first_model]
    fig.add_trace(
        go.Scatter(
            x=actual_one["date"],
            y=actual_one["actual"],
            name=f"Actual {selected}",
            mode="lines+markers",
            line=dict(width=3),
        )
    )
    for model_name in ticker_df["model"].unique():
        m = ticker_df[ticker_df["model"] == model_name]
        fig.add_trace(
            go.Scatter(
                x=m["date"],
                y=m["predicted"],
                name=f"Predicted ({model_name})",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        title=f"{selected}: actual vs predicted (pooled walk-forward)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Per-stock metrics by model")
    per_stock_rows = []
    for csv_file in sorted(PER_STOCK_DIR.glob("*.csv")):
        model_name = csv_file.stem.replace("_", " ").title()
        sdf = pd.read_csv(csv_file).set_index("Ticker")
        if selected in sdf.index:
            row = sdf.loc[selected].to_dict()
            row["Model"] = model_name
            per_stock_rows.append(row)
    if per_stock_rows:
        st.dataframe(
            pd.DataFrame(per_stock_rows).set_index("Model").style.format("{:.3f}"),
            use_container_width=True,
        )

# ---------- Tab 3: training diagnostics ----------
with tab_train:
    st.subheader("Training loss curves (train solid, val dashed)")
    log_y = st.checkbox("Log scale (recommended)", value=True)
    fig = px.line(
        losses_df,
        x="epoch",
        y="loss",
        color="model",
        line_dash="split",
        title="Loss per epoch",
    )
    if log_y:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    val_rows = losses_df[losses_df["split"] == "val"]
    if not val_rows.empty:
        best = (
            val_rows.loc[val_rows.groupby("model")["loss"].idxmin(), ["model", "epoch", "loss"]]
            .rename(columns={"epoch": "best_epoch", "loss": "best_val_mse"})
            .set_index("model")
        )
        st.markdown("##### Best val-MSE epoch per model")
        st.dataframe(best.style.format({"best_val_mse": "{:.6f}"}), use_container_width=True)

# ---------- Tab 4: walk-forward stability ----------
with tab_wf:
    st.subheader("Walk-forward success rate over time")
    fig = px.line(
        wf_df,
        x="window_start",
        y="DirectionalAccuracy",
        color="model",
        markers=True,
        title="Directional accuracy per evaluation window",
    )
    fig.add_hline(y=50, line_dash="dash", line_color="grey")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    stats = (
        wf_df.groupby("model")["DirectionalAccuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "n_windows"})
    )
    st.markdown("##### Stability stats (DirectionalAccuracy across windows)")
    st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
