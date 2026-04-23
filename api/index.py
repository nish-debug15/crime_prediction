import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ml.predict import get_hotspot_prediction

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def dashboard(request: Request):
    """Renders the interactive Crime Heatmap Dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict")
async def api_predict(lat: float, lon: float, hour: int):
    """
    Core API Endpoint: Returns probability and risk status.
    Called by the frontend map when a user clicks a location.
    """
    return get_hotspot_prediction(lat, lon, hour)


@app.get("/explain")
async def explain_model(request: Request):
    """Renders the SHAP Explainability page with the importance plot"""
    return templates.TemplateResponse("explain.html", {"request": request})
