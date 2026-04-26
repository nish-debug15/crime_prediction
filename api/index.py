import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from ml.predict import get_hotspot_prediction

app = FastAPI(title="AI Crime Predictor API", description="Backend Engine for Streamlit")

@app.get("/")
async def root():
    """Health Check Endpoint."""
    return {"message": "Backend API is online and ready.", "status": "Active"}

@app.get("/predict")
async def api_predict(lat: float, lon: float, hour: int):
    """
    Core Engine Endpoint. Streamlit calls this to get predictions.
    Example: /predict?lat=34.05&lon=-118.25&hour=22
    """
    return get_hotspot_prediction(lat, lon, hour)