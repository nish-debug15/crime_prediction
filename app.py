import streamlit as st
import requests
import pandas as pd
import sqlite3
from datetime import datetime
import os

st.set_page_config(page_title="Crime Prediction Radar", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #FFFFFF; font-family: 'Helvetica Neue', sans-serif; }
    .stAlert { background-color: #1E2329; color: #FFFFFF; border: 1px solid #333942; }
    .metric-card {
        background-color: #161A22; 
        border: 1px solid #30363D; 
        padding: 20px; 
        border-radius: 8px; 
        text-align: center;
    }
    .status-critical { color: #FF4B4B; font-weight: bold; font-size: 24px; }
    .status-nominal { color: #00C853; font-weight: bold; font-size: 24px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("SYSTEM OVERRIDE")
    st.markdown("Target coordinates for algorithmic threat assessment.")
    
    lat = st.number_input("Latitude (Y)", value=34.0522, format="%.4f")
    lon = st.number_input("Longitude (X)", value=-118.2437, format="%.4f")
    hour = st.slider("Time Matrix (Hour)", 0, 23, 12)
    
    st.markdown("---")
    st.markdown("**System Specifications:**")
    st.markdown("- Engine: LightGBM Classifier")
    st.markdown("- Architecture: Microservices (FastAPI)")
    st.markdown("- Database: SQLite3 Relational")

st.markdown("<h1>CRIME PREDICTION RADAR</h1>", unsafe_allow_html=True)
st.markdown("`STATUS: ONLINE | ENGINE: FASTAPI | UI: STREAMLIT`")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["[ LIVE RADAR ]", "[ EXPLAINABLE AI ]", "[ AUDIT LOGS ]"])

with tab1:
    st.subheader("Tactical Threat Assessment")
    
    now = datetime.now()
    current_dow = now.weekday()
    current_month = now.month
    current_weekend = 1 if current_dow >= 5 else 0

    API_URL = f"http://127.0.0.1:8000/predict?lat={lat}&lon={lon}&hour={hour}&dow={current_dow}&month={current_month}&is_weekend={current_weekend}"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### System Output")
        try:
            response = requests.get(API_URL)
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == "High Risk":
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #8B949E; margin-bottom: 0px;">THREAT LEVEL</p>
                        <p class="status-critical">HIGH RISK ZONE</p>
                        <h2 style="color: #FF4B4B; margin-top: 0px;">{data['probability'] * 100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #8B949E; margin-bottom: 0px;">THREAT LEVEL</p>
                        <p class="status-nominal">NOMINAL (LOW RISK)</p>
                        <h2 style="color: #00C853; margin-top: 0px;">{data['probability'] * 100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Engine Metadata (Model V1.0):**")
                st.json(data['model_metadata'])

            elif response.status_code == 422:
                st.error("API Error 422: Parameter mismatch. The backend requires 6 features but Streamlit sent the wrong amount.")
            else:
                st.error(f"API Connection Failed. Status Code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("CRITICAL ERROR: Cannot connect to FastAPI engine. Ensure `uvicorn api.index:app` is running on Port 8000.")

    with col2:
        st.markdown("### Geospatial Target")
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        st.map(map_df, zoom=12)


with tab2:
    st.subheader("Algorithmic Transparency (SHAP)")
    st.markdown("Law enforcement systems cannot utilize Black Box AI. Below is the mathematical breakdown of feature weights driving the current predictive model.")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        if os.path.exists("static/shap_summary.png"):
            st.image("static/shap_summary.png", caption="SHAP Feature Importance (High Risk Drivers)")
        else:
            st.warning("SHAP plot not found. Run `ml/train.py` to generate.")
            
    with col_img2:
        if os.path.exists("static/confusion_matrix.png"):
            st.image("static/confusion_matrix.png", caption="Model Evaluation: Confusion Matrix")
        else:
            st.warning("Confusion Matrix not found. Run `ml/train.py` to generate.")


with tab3:
    st.subheader("Real-Time Audit Database")
    st.markdown("All API requests are permanently logged to `predictions.db` to ensure strict algorithmic accountability.")
    
    def load_audit_logs():
        db_path = "data/predictions.db"
        if not os.path.exists(db_path):
            return pd.DataFrame(columns=["Notice"])
            
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT timestamp, lat, lon, hour, probability, status FROM logs ORDER BY id DESC LIMIT 50", conn)
            conn.close()
            
            if not df.empty:
                df['probability'] = (df['probability'] * 100).round(1).astype(str) + '%'
            return df
        except Exception as e:
            return pd.DataFrame([{"Error": str(e)}])

    logs_df = load_audit_logs()
    
    if logs_df.empty or "Notice" in logs_df.columns:
        st.info("No audit logs found. The database will generate upon the first successful API scan.")
    else:
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
        
        if st.button("Refresh Audit Logs"):
            st.rerun()