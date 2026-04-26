import os
import sqlite3
from datetime import datetime
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../models/hotspot_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "../data/predictions.db")

def init_db():
    """
    Initializes the SQLite Database. 
    Creates the 'logs' table automatically if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            lat REAL,
            lon REAL,
            hour INTEGER,
            is_hotspot BOOLEAN,
            probability REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction_sql(lat, lon, hour, result):
    """
    Saves the prediction request and result to the SQLite database.
    Uses parameterized queries (?, ?) to prevent SQL injection.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO logs (timestamp, lat, lon, hour, is_hotspot, probability, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        lat, 
        lon, 
        hour, 
        result["is_hotspot"], 
        result["probability"], 
        result["status"]
    ))
    conn.commit()
    conn.close()

init_db()

def get_hotspot_prediction(lat, lon, hour):
    try:
        model = joblib.load(MODEL_PATH)
        input_data = pd.DataFrame([[lat, lon, hour]], columns=['lat_grid', 'lon_grid', 'hour'])
        
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
        
        log_prediction_sql(lat, lon, hour, result)
        
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test a prediction
    print(get_hotspot_prediction(34.04, -118.26, 22))
    print(f"Prediction saved to SQLite Database at: {DB_PATH}")