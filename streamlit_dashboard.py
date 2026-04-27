import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Crime Prediction Dashboard", layout="wide")
st.title("🚔 Crime Prediction Dashboard")

st.sidebar.header("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Districts", df['District'].nunique() if 'District' in df.columns else 0)
    with col3:
        st.metric("Crime Types", df['CrimeType'].nunique() if 'CrimeType' in df.columns else 0)
    
    # Crime Distribution Chart
    st.subheader("Crime Distribution")
    if 'CrimeType' in df.columns:
        fig = px.bar(df['CrimeType'].value_counts(), title="Crimes by Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Analysis
    st.subheader("Crime Trends Over Time")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        crime_by_date = df.groupby('Date').size()
        st.line_chart(crime_by_date)
    
    # Geographic Distribution
    st.subheader("Geographic Analysis")
    if 'District' in df.columns:
        district_crimes = df['District'].value_counts()
        fig = px.pie(values=district_crimes.values, names=district_crimes.index, title="Crimes by District")
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw Data Table
    st.subheader("Raw Data")
    st.dataframe(df)
    
else:
    st.info("📊 Upload a CSV file to get started with crime prediction analysis")