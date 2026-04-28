import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from ml.predict import get_hotspot_prediction

app = FastAPI(title="AI Predictive Policing API", description="Microservices REST Backend")

@app.get("/")
async def root():
    """Health Check Endpoint"""
    return {"message": "Backend API is online and ready.", "status": "Active"}

@app.get("/predict")
async def api_predict(lat: float, lon: float, hour: int, dow: int, month: int, is_weekend: int):
    """
    Core Engine Endpoint.
    Accepts all 6 spatial-temporal features and routes them to the LightGBM model.
    """
    return get_hotspot_prediction(lat, lon, hour, dow, month, is_weekend)