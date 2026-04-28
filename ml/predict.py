import os
import sqlite3
from datetime import datetime
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/hotspot_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "../data/predictions.db")

def init_db():
    """Initializes the SQLite Database with the complete 6-feature schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            lat REAL,
            lon REAL,
            hour INTEGER,
            day_of_week INTEGER,
            month INTEGER,
            is_weekend INTEGER,
            is_hotspot BOOLEAN,
            probability REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction_sql(lat, lon, hour, day_of_week, month, is_weekend, result):
    """Securely logs the 6-feature prediction to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO logs (timestamp, lat, lon, hour, day_of_week, month, is_weekend, is_hotspot, probability, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            lat, lon, hour, day_of_week, month, is_weekend,
            result["is_hotspot"],
            result["probability"],
            result["status"]
        ))
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"Database Schema Error: {e}")
        print("CRITICAL FIX: You must delete your existing 'data/predictions.db' file. The database schema has upgraded to include dates, and needs to be rebuilt.")

init_db()

def get_hotspot_prediction(lat, lon, hour, day_of_week, month, is_weekend):
    try:
        model = joblib.load(MODEL_PATH)
        
        input_data = pd.DataFrame(
            [[lat, lon, hour, day_of_week, month, is_weekend]],
            columns=['lat_grid', 'lon_grid', 'hour', 'day_of_week', 'month', 'is_weekend']
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = {
            "is_hotspot": bool(prediction),
            "probability": round(float(probability), 4),
            "status": "High Risk" if prediction == 1 else "Low Risk",
            "model_metadata": {
                "accuracy": "89.91%",
                "precision": "0.85",
                "recall": "0.71",
                "f1_score": "0.78"
            }
        }

        log_prediction_sql(lat, lon, hour, day_of_week, month, is_weekend, result)
        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    from datetime import datetime as dt
    now = dt.now()
    result = get_hotspot_prediction(
        lat=34.04,
        lon=-118.26,
        hour=22,
        day_of_week=now.weekday(),
        month=now.month,
        is_weekend=1 if now.weekday() >= 5 else 0
    )
    print(f"Prediction Output: {result}")
    print(f"Saved to: {DB_PATH}")