import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Proactive Crime Predictor", layout="wide")

st.sidebar.title("Control Panel")
st.sidebar.markdown("Configure parameters for spatial-temporal risk assessment.")

lat_input = st.sidebar.number_input("Latitude", value=34.0522, format="%.4f")
lon_input = st.sidebar.number_input("Longitude", value=-118.2437, format="%.4f")
hour_input = st.sidebar.slider("Hour of Day (Military)", min_value=0, max_value=23, value=12)

st.title("Proactive Crime Prediction Dashboard")
st.markdown("Interfacing with FastAPI Microservice for real-time algorithmic risk assessment.")

tab1, tab2, tab3 = st.tabs(["Live Prediction", "Explainable AI", "System Status"])

with tab1:
    if st.sidebar.button("Run Prediction", type="primary"):
        with st.spinner("Querying ML Backend..."):
            try:
                res = requests.get(f"http://127.0.0.1:8000/predict?lat={lat_input}&lon={lon_input}&hour={hour_input}")
                data = res.json()
                
                if "error" in data:
                    st.error(f"Model Error: {data['error']}")
                else:
                    st.markdown("### Prediction Results")
                    col1, col2, col3 = st.columns(3)
                    
                    status = data.get("status", "Unknown")
                    prob = data.get("probability", 0) * 100
                    accuracy = data.get("model_metadata", {}).get("accuracy", "N/A")
                    
                    col1.metric("Risk Status", status)
                    col2.metric("Probability", f"{prob:.1f}%")
                    col3.metric("System Accuracy", accuracy)
                    
                    st.markdown("### Target Location")
                    map_data = pd.DataFrame({'lat': [lat_input], 'lon': [lon_input]})
                    st.map(map_data, zoom=12)
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Ensure the FastAPI backend is running on port 8000.")
    else:
        st.info("Adjust parameters in the sidebar and click 'Run Prediction' to initialize the model.")

with tab2:
    st.subheader("Model Explainability (SHAP)")
    st.markdown("Analyzing feature importance and algorithmic decision pathways to ensure transparent forecasting.")
    try:
        st.image("static/shap_summary.png", use_container_width=True)
    except FileNotFoundError:
        st.warning("SHAP image not found. Ensure the ML pipeline has executed and generated the required visual artifacts.")

with tab3:
    st.subheader("Microservices Architecture")
    st.markdown("""
    * **Frontend UI:** Streamlit
    * **Backend Engine:** FastAPI (REST API)
    * **Machine Learning Model:** LightGBM Binary Classifier
    * **Audit Logging:** SQLite Relational Database
    * **Data Preprocessing:** Pandas & Scikit-Learn
    """)
    st.success("Frontend Application Running Successfully on Port 8501")