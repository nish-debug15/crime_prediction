import streamlit as st
import pandas as pd
import sqlite3
import sys
import os
from datetime import datetime, date

# ── Path setup so `ml.predict` resolves regardless of CWD ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.predict import get_hotspot_prediction

# ── Pydeck (bundled with Streamlit) ────────────────────────────────────────
import pydeck as pdk

# ── SQLite path (same as predict.py) ───────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.db")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CrimeSight | LA Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0c10;
    color: #e2e8f0;
}
.block-container { padding-top: 1.5rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2330;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #13171f;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 1rem;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* High risk banner */
.risk-high {
    background: linear-gradient(135deg, #3d0000 0%, #1a0000 100%);
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.risk-low {
    background: linear-gradient(135deg, #00200a 0%, #001008 100%);
    border: 1px solid #00cc55;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #8899aa; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def derive_temporal(selected_date: date, hour: int):
    """Return day_of_week, month, is_weekend from date + hour."""
    dow = selected_date.weekday()          # 0=Mon … 6=Sun
    month = selected_date.month
    is_weekend = 1 if dow >= 5 else 0
    return dow, month, is_weekend


def load_logs(limit: int = 200) -> pd.DataFrame:
    """Pull recent prediction logs from SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            f"SELECT * FROM logs ORDER BY id DESC LIMIT {limit}", conn
        )
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()


def hourly_risk_sweep(lat: float, lon: float, dow: int, month: int, is_weekend: int) -> pd.DataFrame:
    """Run prediction for all 24 hours at given location/date."""
    rows = []
    for h in range(24):
        r = get_hotspot_prediction(lat, lon, h, dow, month, is_weekend)
        rows.append({
            "hour": h,
            "probability": r.get("probability", 0),
            "status": r.get("status", "Unknown"),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔭 CrimeSight")
    st.markdown('<p class="mono">LA Predictive Risk Intelligence</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**📍 Location**")
    lat_input = st.number_input("Latitude", value=34.0522, format="%.4f", step=0.001)
    lon_input = st.number_input("Longitude", value=-118.2437, format="%.4f", step=0.001)

    st.markdown("**🕐 Time**")
    date_input = st.date_input("Date", value=date.today())
    hour_input = st.slider("Hour of Day (0–23)", min_value=0, max_value=23, value=datetime.now().hour)

    # Auto-derive temporal features
    dow, month, is_weekend = derive_temporal(date_input, hour_input)

    st.markdown('<p class="mono">'
                f'day_of_week={dow} &nbsp; month={month} &nbsp; is_weekend={is_weekend}'
                '</p>', unsafe_allow_html=True)

    st.divider()
    run_btn = st.button("⚡ Run Prediction", type="primary", use_container_width=True)
    sweep_btn = st.button("📊 Hourly Risk Sweep", use_container_width=True)

    st.divider()
    st.markdown('<p class="mono">Model: LightGBM · Acc 92.86%<br>High-Risk F1: 0.85</p>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["⚡ Live Prediction", "🧠 Explainable AI", "🗄️ Audit Logs"])


# ─── TAB 1 · LIVE PREDICTION ───────────────────────────────────────────────
with tab1:
    st.markdown("### Live Risk Assessment")

    # ── Session state for result persistence ───────────────────────────────
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "hourly_df" not in st.session_state:
        st.session_state.hourly_df = None

    # ── Run single prediction ──────────────────────────────────────────────
    if run_btn:
        with st.spinner("Running model inference…"):
            result = get_hotspot_prediction(lat_input, lon_input, hour_input, dow, month, is_weekend)
        st.session_state.last_result = result
        st.session_state.hourly_df = None  # clear old sweep

    # ── Hourly sweep ───────────────────────────────────────────────────────
    if sweep_btn:
        with st.spinner("Sweeping 24 hours…"):
            hdf = hourly_risk_sweep(lat_input, lon_input, dow, month, is_weekend)
        st.session_state.hourly_df = hdf
        st.session_state.last_result = None

    # ── Display single result ──────────────────────────────────────────────
    result = st.session_state.last_result
    if result:
        if "error" in result:
            st.error(f"Model error: {result['error']}")
        else:
            status = result["status"]
            prob = result["probability"] * 100
            css_cls = "risk-high" if status == "High Risk" else "risk-low"
            icon = "🔴" if status == "High Risk" else "🟢"

            st.markdown(
                f'<div class="{css_cls}">'
                f'<h2 style="margin:0">{icon} {status}</h2>'
                f'<p class="mono" style="margin:4px 0 0">Probability: {prob:.1f}% &nbsp;|&nbsp; '
                f'lat={lat_input} lon={lon_input} hour={hour_input:02d}h</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk Probability", f"{prob:.1f}%")
            c2.metric("Accuracy", result["model_metadata"]["accuracy"])
            c3.metric("Precision", result["model_metadata"]["precision"])
            c4.metric("F1 Score", result["model_metadata"]["f1_score"])

            # ── Pydeck heatmap ────────────────────────────────────────────
            st.markdown("#### 🗺️ Risk Heatmap")

            # Build a small grid of nearby points weighted by probability
            import numpy as np
            rng = np.random.default_rng(42)
            n = 80
            lats = rng.normal(lat_input, 0.015, n)
            lons = rng.normal(lon_input, 0.015, n)
            weights = rng.beta(2 if status == "High Risk" else 1,
                               1 if status == "High Risk" else 2, n)
            heat_df = pd.DataFrame({"lat": lats, "lon": lons, "weight": weights})

            # Add actual prediction point
            heat_df = pd.concat([
                heat_df,
                pd.DataFrame({"lat": [lat_input], "lon": [lon_input], "weight": [result["probability"]]})
            ], ignore_index=True)

            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=heat_df,
                get_position=["lon", "lat"],
                get_weight="weight",
                radius_pixels=60,
                intensity=1,
                threshold=0.05,
                color_range=[
                    [0, 128, 0, 80],
                    [255, 255, 0, 120],
                    [255, 100, 0, 160],
                    [220, 0, 0, 200],
                ],
            )

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame({"lat": [lat_input], "lon": [lon_input]}),
                get_position=["lon", "lat"],
                get_radius=200,
                get_fill_color=[255, 80, 80, 220] if status == "High Risk" else [0, 220, 100, 220],
                pickable=True,
            )

            view = pdk.ViewState(latitude=lat_input, longitude=lon_input, zoom=12, pitch=40)
            st.pydeck_chart(pdk.Deck(
                layers=[heatmap_layer, scatter_layer],
                initial_view_state=view,
                map_style="mapbox://styles/mapbox/dark-v10",
            ))

    # ── Display hourly sweep ───────────────────────────────────────────────
    hdf = st.session_state.hourly_df
    if hdf is not None:
        st.markdown("#### 📊 Hourly Risk Profile (0–23h)")
        st.markdown(f'<p class="mono">lat={lat_input} lon={lon_input} · {date_input.strftime("%a %d %b %Y")}</p>',
                    unsafe_allow_html=True)

        # Color bars: red=high, green=low
        hdf["color"] = hdf["status"].apply(
            lambda s: "#ff4444" if s == "High Risk" else "#00cc55"
        )

        # Use Streamlit bar chart via Altair for color support
        import altair as alt
        chart = (
            alt.Chart(hdf)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("hour:O", title="Hour of Day", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("probability:Q", title="Risk Probability", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("status:N", scale=alt.Scale(
                    domain=["High Risk", "Low Risk"],
                    range=["#ff4444", "#00cc55"]
                )),
                tooltip=["hour", alt.Tooltip("probability:Q", format=".2%"), "status"],
            )
            .properties(height=320)
            .configure_view(strokeOpacity=0)
            .configure_axis(
                gridColor="#1e2330",
                labelColor="#8899aa",
                titleColor="#8899aa",
                labelFont="IBM Plex Mono",
                titleFont="IBM Plex Mono",
            )
            .configure_legend(
                labelColor="#e2e8f0",
                titleColor="#8899aa",
                labelFont="IBM Plex Mono",
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # Peak risk hour callout
        peak = hdf.loc[hdf["probability"].idxmax()]
        st.info(f"⚠️ Peak risk: **{int(peak['hour']):02d}:00h** — {peak['probability']:.1%} probability")

    if not run_btn and not sweep_btn and result is None and hdf is None:
        st.info("← Configure parameters in sidebar, then click **Run Prediction** or **Hourly Risk Sweep**.")


# ─── TAB 2 · EXPLAINABLE AI ────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Explainability")

    col_shap, col_info = st.columns([2, 1])

    with col_shap:
        st.markdown("#### Global Feature Importance (SHAP Summary)")
        shap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "shap_summary.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True)
        else:
            st.warning("SHAP image not found at `static/shap_summary.png`. Run ML pipeline first.")

    with col_info:
        st.markdown("#### Feature Reference")
        feat_df = pd.DataFrame({
            "Feature": ["lat_grid", "lon_grid", "hour", "day_of_week", "month", "is_weekend"],
            "Type": ["spatial", "spatial", "temporal", "temporal", "temporal", "temporal"],
            "Range": ["continuous", "continuous", "0–23", "0–6", "1–12", "0/1"],
        })
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

        st.markdown("#### Model Card")
        st.markdown("""
| Metric | Value |
|--------|-------|
| Algorithm | LightGBM |
| Accuracy | 92.86% |
| High-Risk F1 | 0.85 |
| Features | 6 |
| Target | `is_hotspot` |
""")

    st.divider()
    st.markdown("#### Per-Prediction SHAP Waterfall")
    result = st.session_state.get("last_result")
    if result and "error" not in result:
        st.markdown("Approximate SHAP waterfall based on last prediction inputs.")
        try:
            import joblib, shap
            import matplotlib.pyplot as plt
            import numpy as np

            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hotspot_model.pkl")
            model = joblib.load(model_path)
            input_df = pd.DataFrame(
                [[lat_input, lon_input, hour_input, dow, month, is_weekend]],
                columns=['lat_grid', 'lon_grid', 'hour', 'day_of_week', 'month', 'is_weekend']
            )
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

            feat_names = ['lat_grid', 'lon_grid', 'hour', 'day_of_week', 'month', 'is_weekend']
            feat_values = [lat_input, lon_input, hour_input, dow, month, is_weekend]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0a0c10')
            ax.set_facecolor('#13171f')
            colors = ['#ff4444' if v > 0 else '#00cc55' for v in vals]
            bars = ax.barh(feat_names, vals, color=colors, edgecolor='none', height=0.5)
            for bar, val, fval in zip(bars, vals, feat_values):
                ax.text(val + (0.003 if val >= 0 else -0.003),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}  [{fval}]",
                        va='center', ha='left' if val >= 0 else 'right',
                        color='#e2e8f0', fontsize=8, fontfamily='monospace')
            ax.axvline(0, color='#3a4455', linewidth=1)
            ax.set_xlabel("SHAP value (impact on log-odds)", color='#8899aa', fontsize=8)
            ax.tick_params(colors='#8899aa', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e2330')
            ax.set_title("SHAP Waterfall — Last Prediction", color='#e2e8f0', fontsize=10, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except ImportError:
            st.info("Install `shap` for per-prediction waterfall: `pip install shap`")
        except Exception as e:
            st.warning(f"Waterfall unavailable: {e}")
    else:
        st.info("Run a single prediction first (Tab 1 → **Run Prediction**) to see per-prediction SHAP.")


# ─── TAB 3 · AUDIT LOGS ────────────────────────────────────────────────────
with tab3:
    st.markdown("### Prediction Audit Log")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        log_limit = st.selectbox("Show last N rows", [50, 100, 200, 500], index=1)
    with col_ctrl2:
        filter_status = st.selectbox("Filter by status", ["All", "High Risk", "Low Risk"])
    with col_ctrl3:
        refresh_btn = st.button("🔄 Refresh Logs", use_container_width=True)

    logs_df = load_logs(log_limit)

    if logs_df.empty:
        st.info("No predictions logged yet. Run a prediction to populate this table.")
    else:
        if filter_status != "All":
            logs_df = logs_df[logs_df["status"] == filter_status]

        # ── Summary metrics ────────────────────────────────────────────────
        if not logs_df.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Predictions", len(logs_df))
            high_pct = (logs_df["status"] == "High Risk").mean() * 100
            m2.metric("High Risk %", f"{high_pct:.1f}%")
            m3.metric("Avg Probability", f"{logs_df['probability'].mean():.1%}")
            m4.metric("Latest", logs_df["timestamp"].iloc[0][:16] if "timestamp" in logs_df.columns else "—")

            st.divider()

            # ── Styled dataframe ───────────────────────────────────────────
            display_cols = ["timestamp", "lat", "lon", "hour", "day_of_week",
                            "month", "is_weekend", "probability", "status"]
            display_df = logs_df[[c for c in display_cols if c in logs_df.columns]].copy()
            display_df["probability"] = display_df["probability"].apply(lambda x: f"{x:.1%}")

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "status": st.column_config.TextColumn("Status"),
                    "probability": st.column_config.TextColumn("Probability"),
                    "timestamp": st.column_config.TextColumn("Timestamp"),
                }
            )

            # ── Log trend sparkline ────────────────────────────────────────
            if len(logs_df) > 3:
                st.markdown("#### Probability Trend (last predictions)")
                import altair as alt
                trend_df = logs_df[["timestamp", "probability", "status"]].copy()
                trend_df = trend_df.iloc[::-1].reset_index(drop=True)
                trend_df["index"] = trend_df.index

                trend_chart = (
                    alt.Chart(trend_df)
                    .mark_line(point=True, strokeWidth=2)
                    .encode(
                        x=alt.X("index:Q", title="Prediction #", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("status:N", scale=alt.Scale(
                            domain=["High Risk", "Low Risk"],
                            range=["#ff4444", "#00cc55"]
                        )),
                        tooltip=["timestamp", alt.Tooltip("probability:Q", format=".2%"), "status"],
                    )
                    .properties(height=200)
                    .configure_view(strokeOpacity=0)
                    .configure_axis(
                        gridColor="#1e2330", labelColor="#8899aa",
                        titleColor="#8899aa", labelFont="IBM Plex Mono",
                    )
                    .configure_legend(labelColor="#e2e8f0", labelFont="IBM Plex Mono")
                )
                st.altair_chart(trend_chart, use_container_width=True)