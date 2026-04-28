import streamlit as st
import requests
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="Crime Prediction Radar", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark Tactical Theme */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* Action Button */
    .stButton>button {
        width: 100%;
        background-color: #0a0a0a !important;
        color: #00ffcc !important;
        border: 1px solid #00ffcc !important;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.2);
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: 0.3s all ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00ffcc !important;
        color: #000000 !important;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.6);
    }
    
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #111111;
        border-left: 4px solid #00ffcc;
        padding: 15px 20px;
        border-radius: 4px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def fetch_audit_logs():
    """Queries the SQLite database to show the live logging system."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.db")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT timestamp, lat, lon, hour, probability, status FROM logs ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

st.sidebar.markdown("## SYSTEM OVERRIDE")
st.sidebar.markdown("Target coordinates for algorithmic threat assessment.")

lat_input = st.sidebar.number_input("Latitude (Y)", value=34.0522, format="%.4f")
lon_input = st.sidebar.number_input("Longitude (X)", value=-118.2437, format="%.4f")
hour_input = st.sidebar.slider("Time Matrix (Hour)", min_value=0, max_value=23, value=12)

st.sidebar.divider()
st.sidebar.markdown("""
**System Specifications:**
* Engine: LightGBM Classifier
* Architecture: Microservices (FastAPI + Streamlit)
* Database: SQLite3 Relational
""")

st.markdown("# CRIME PREDICTION RADAR")
st.markdown("`STATUS: ONLINE | ENGINE: FASTAPI | UI: STREAMLIT`")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["[ LIVE RADAR ]", "[ EXPLAINABLE AI ]", "[ AUDIT LOGS ]"])

with tab1:
    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        run_scan = st.button("INITIATE PREDICTIVE SCAN")

    target_color = '#00ffccff' 
    map_zoom = 11

    if run_scan:
        with st.spinner("Connecting to Backend API (Port 8000)..."):
            try:
                res = requests.get(f"http://127.0.0.1:8000/predict?lat={lat_input}&lon={lon_input}&hour={hour_input}")
                data = res.json()
                
                if "error" in data:
                    st.error(f"SYSTEM FAULT: {data['error']}")
                else:
                    st.markdown("### THREAT ASSESSMENT REPORT")
                    m1, m2, m3 = st.columns(3)
                    
                    if data.get("is_hotspot"):
                        st.error("[ALERT] HIGH RISK ZONE DETECTED")
                        m1.metric("THREAT LEVEL", "CRITICAL", delta="Action Required", delta_color="inverse")
                        target_color = '#ff0000ff' 
                    else:
                        st.success("[SECURE] NOMINAL RISK DETECTED")
                        m1.metric("THREAT LEVEL", "NOMINAL", delta="Stable", delta_color="normal")
                        target_color = '#00ff00ff' 
                        
                    m2.metric("PROBABILITY INDEX", f"{data.get('probability', 0) * 100:.1f}%")
                    m3.metric("ENGINE ACCURACY", data.get("model_metadata", {}).get("accuracy", "89.9%"))
                    map_zoom = 13
                    
            except requests.exceptions.ConnectionError:
                st.error("[FATAL ERROR] Backend Offline. Ensure FastAPI server is running on Port 8000.")
    else:
        st.info("Awaiting parameters. Adjust the sidebar and initiate scan.")

    st.markdown("### GEOSPATIAL RADAR")
    target_df = pd.DataFrame([{'lat': lat_input, 'lon': lon_input, 'color': target_color, 'size': 1200}])
    st.map(target_df, latitude='lat', longitude='lon', color='color', size='size', zoom=map_zoom)

with tab2:
    st.markdown("### SHAP Feature Importance")
    st.markdown("Algorithmic transparency protocol. The chart below reveals the mathematical weights driving the model's spatial-temporal assessment.")
    try:
        st.image("static/shap_summary.png", use_container_width=True)
    except FileNotFoundError:
        st.warning("[WARNING] SHAP artifact not found. Ensure `shap_summary.png` is located in the `static/` directory.")

with tab3:
    st.markdown("### Real-Time Audit Database")
    st.markdown("All API requests are permanently logged to `predictions.db` to ensure strict algorithmic accountability.")
    
    logs_df = fetch_audit_logs()
    
    if not logs_df.empty:
        logs_df["probability"] = logs_df["probability"].apply(lambda x: f"{x:.1%}")
        st.dataframe(
            logs_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "status": st.column_config.TextColumn("Risk Status"),
                "probability": st.column_config.TextColumn("Probability"),
                "timestamp": st.column_config.TextColumn("Timestamp")
            }
        )
    else:
        st.info("Log registry empty. Execute a scan in the Live Radar to populate the database.")