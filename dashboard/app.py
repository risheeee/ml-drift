import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DB_PATH = Path("data/drift_store.db")
WINDOWS_PATH = Path("data/windows")
MONITOR_FEATURES = ["V1", "V2", "V3", "V4", "V14", "V17", "Amount"]

st.set_page_config(page_title="Drift Monitor", layout="wide")


@st.cache_data(ttl=30)
def load_drift_metrics():
    if not DB_PATH.exists(): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM drift_metrics ORDER BY window_id, feature", conn)
    conn.close(); return df


@st.cache_data(ttl=30)
def load_performance_metrics():
    if not DB_PATH.exists(): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM model_performance ORDER BY window_id", conn)
    conn.close(); return df


@st.cache_data
def load_window(window_id: int):
    path = WINDOWS_PATH / f"window_{window_id}" / "data.parquet"
    return pd.read_parquet(path) if path.exists() else None


st.title("ML Drift Pipeline — Monitoring Dashboard")
st.caption("Credit Card Fraud Detection · Self-Healing Production Pipeline")

drift_df = load_drift_metrics()
perf_df = load_performance_metrics()

if drift_df.empty:
    st.warning("No metrics found. Run `python src/drift.py` to populate the database.")
    st.stop()

# ── KPI Row ────────────────────────────────────────────────────────────────────
psi_df = drift_df[drift_df["metric_type"] == "psi"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Window", int(psi_df["window_id"].max()) if not psi_df.empty else 0, delta="of 6 total")
c2.metric("Windows Triggered Retraining", int(drift_df[drift_df["triggered"]==1]["window_id"].nunique()))
c3.metric("Avg PSI", f"{psi_df['value'].mean():.3f}" if not psi_df.empty else "N/A", delta="threshold: 0.20")
if not perf_df.empty and perf_df["auprc"].notna().any():
    c4.metric("Latest AUPRC", f"{perf_df.iloc[-1]['auprc']:.4f}")
st.divider()

# ── PSI Heatmap ────────────────────────────────────────────────────────────────
st.subheader("Feature Drift (PSI) by Window")
st.caption("PSI > 0.20 = significant drift (retrain) | PSI > 0.10 = moderate (monitor)")
psi_pivot = (psi_df[psi_df["feature"].isin(MONITOR_FEATURES)]
             .pivot_table(index="feature", columns="window_id", values="value", aggfunc="mean").round(4))
if not psi_pivot.empty:
    fig = go.Figure(go.Heatmap(
        z=psi_pivot.values, x=[f"Window {c}" for c in psi_pivot.columns], y=psi_pivot.index.tolist(),
        colorscale=[[0,"#2d6a4f"],[0.10,"#74c69d"],[0.20,"#f4a261"],[0.40,"#e63946"],[1.0,"#6d023b"]],
        zmin=0, zmax=0.5, text=psi_pivot.values.round(3), texttemplate="%{text}",
        colorbar=dict(title="PSI", tickvals=[0,0.10,0.20,0.40],
                      ticktext=["0 (stable)","0.10 (watch)","0.20 (retrain)","0.40 (severe)"])))
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── PSI Trend ─────────────────────────────────────────────────────────────────
st.subheader("PSI Trend Per Feature")
selected = st.multiselect("Features", MONITOR_FEATURES, default=["V1","V2","V14","Amount"])
if selected and not psi_df.empty:
    fig2 = go.Figure()
    for feat in selected:
        d = psi_df[psi_df["feature"]==feat].sort_values("window_id")
        fig2.add_trace(go.Scatter(x=d["window_id"], y=d["value"], name=feat, mode="lines+markers", marker=dict(size=8)))
    fig2.add_hline(y=0.20, line_dash="dash", line_color="orange", annotation_text="Retrain (0.20)")
    fig2.add_hline(y=0.10, line_dash="dot",  line_color="gray",   annotation_text="Watch (0.10)")
    fig2.update_layout(height=320, xaxis=dict(title="Window", tickmode="linear", dtick=1),
                       yaxis_title="PSI", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig2, use_container_width=True)
st.divider()

# ── Model Performance ──────────────────────────────────────────────────────────
st.subheader("Model Performance Over Windows")
if not perf_df.empty and perf_df["auprc"].notna().any():
    fig3 = go.Figure()
    if perf_df["auprc"].notna().any():
        fig3.add_trace(go.Scatter(x=perf_df["window_id"], y=perf_df["auprc"], name="AUPRC",
                                  mode="lines+markers", line=dict(color="#4361ee")))
    if perf_df["recall_at_5pct_fpr"].notna().any():
        fig3.add_trace(go.Scatter(x=perf_df["window_id"], y=perf_df["recall_at_5pct_fpr"],
                                  name="Recall@5%FPR", mode="lines+markers", line=dict(color="#f77f00")))
    if perf_df["score_dist_psi"].notna().any():
        fig3.add_trace(go.Scatter(x=perf_df["window_id"], y=perf_df["score_dist_psi"],
                                  name="Score PSI", mode="lines+markers", line=dict(color="#e63946", dash="dot")))
    fig3.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="AUPRC floor")
    fig3.update_layout(height=320, xaxis=dict(title="Window", tickmode="linear", dtick=1),
                       margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig3, use_container_width=True)
st.divider()

# ── Distribution Comparison ────────────────────────────────────────────────────
st.subheader("Feature Distribution — Reference vs Selected Window")
col1, col2 = st.columns(2)
feat = col1.selectbox("Feature", MONITOR_FEATURES)
win  = col2.selectbox("Compare Window", list(range(1, 7)), index=3)
ref_df = load_window(1); curr_df = load_window(win)
if ref_df is not None and curr_df is not None and feat in ref_df.columns:
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=ref_df[feat].clip(-5,5), name="Window 1 (Reference)",
                                opacity=0.6, nbinsx=50, histnorm="probability density", marker_color="#4361ee"))
    fig4.add_trace(go.Histogram(x=curr_df[feat].clip(-5,5), name=f"Window {win}",
                                opacity=0.6, nbinsx=50, histnorm="probability density", marker_color="#f77f00"))
    fig4.update_layout(barmode="overlay", height=280, xaxis_title=feat, margin=dict(l=0,r=0,t=20,b=0))
    if feat in psi_pivot.index and win in psi_pivot.columns:
        psi_val = psi_pivot.loc[feat, win]
        status = "🔴 Drifted" if psi_val > 0.20 else "🟡 Watch" if psi_val > 0.10 else "🟢 Stable"
        st.caption(f"PSI = **{psi_val:.4f}** {status}")
    st.plotly_chart(fig4, use_container_width=True)
st.divider()

# ── Drift Events Log ───────────────────────────────────────────────────────────
st.subheader("Drift Events Log")
triggered = drift_df[drift_df["triggered"]==1].sort_values(["window_id","feature"])
if not triggered.empty:
    st.dataframe(triggered[["window_id","feature","metric_type","value","threshold","computed_at"]]
                 .rename(columns={"window_id":"Window","feature":"Feature","metric_type":"Metric",
                                   "value":"Value","threshold":"Threshold","computed_at":"Detected At"}),
                 use_container_width=True, hide_index=True)
else:
    st.success("No drift events triggered yet.")

if st.button("Refresh"): st.cache_data.clear(); st.rerun()