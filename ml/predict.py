import joblib
import pandas as pd
import os
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/hotspot_model.pkl")
LOG_PATH = os.path.join(os.path.dirname(__file__), "../data/prediction_logs.csv")

def log_prediction(lat, lon, hour, result):
    """
    Saves the prediction request and result to a CSV file for auditing.
    """
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lat": lat,
        "lon": lon,
        "hour": hour,
        "is_hotspot": result["is_hotspot"],
        "probability": result["probability"],
        "status": result["status"]
    }
    
    df = pd.DataFrame([log_entry])
    
    file_exists = os.path.isfile(LOG_PATH)
    df.to_csv(LOG_PATH, mode='a', index=False, header=not file_exists)

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
        
        log_prediction(lat, lon, hour, result)
        
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print(get_hotspot_prediction(34.04, -118.26, 22))
    print(f"Prediction saved to {LOG_PATH}")