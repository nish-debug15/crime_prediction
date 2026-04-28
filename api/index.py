import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from ml.predict import get_hotspot_prediction

app = FastAPI(
    title="AI Crime Predictor API", description="Backend Engine for Streamlit"
)


@app.get("/")
async def root():
    return {"message": "Backend API is online and ready.", "status": "Active"}


@app.get("/predict")
async def api_predict(
    lat: float, lon: float, hour: int, day_of_week: int, month: int, is_weekend: int
):
    """
    Core prediction endpoint.
    Example: /predict?lat=34.05&lon=-118.25&hour=22&day_of_week=1&month=4&is_weekend=0
    """
    return get_hotspot_prediction(lat, lon, hour, day_of_week, month, is_weekend)


@app.get("/history")
async def api_history(limit: int = 50):
    """Returns last N predictions from SQLite."""
    import sqlite3
    import pandas as pd

    DB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data/predictions.db"
    )
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM logs ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df.to_dict(orient="records")


@app.get("/stats")
async def api_stats():
    """Aggregate stats on all predictions made."""
    import sqlite3
    import pandas as pd

    DB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data/predictions.db"
    )
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()
    return {
        "total_predictions": len(df),
        "hotspot_count": int(df["is_hotspot"].sum()),
        "hotspot_pct": round(float(df["is_hotspot"].mean()) * 100, 2),
        "avg_probability": round(float(df["probability"].mean()), 4),
    }
