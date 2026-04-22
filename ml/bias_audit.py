import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

def run_bias_check():
    print("Running Fairness Audit...")
    
    model = joblib.load("models/hotspot_model.pkl")
    df = pd.read_csv("data/model_features.csv")
    
    median_lat = df['lat_grid'].median()
    df['is_north'] = (df['lat_grid'] > median_lat).astype(int)
    
    X = df[['lat_grid', 'lon_grid', 'hour']]
    df['preds'] = model.predict(X)
    
    north_rate = df[df['is_north'] == 1]['preds'].mean()
    south_rate = df[df['is_north'] == 0]['preds'].mean()
    
    disparate_impact = south_rate / north_rate if north_rate > 0 else 1
    
    print(f"North LA Hotspot Rate: {north_rate:.2%}")
    print(f"South LA Hotspot Rate: {south_rate:.2%}")
    print(f"Disparate Impact Ratio: {disparate_impact:.4f}")
    
    if 0.8 < disparate_impact < 1.25:
        print("Fairness Check Passed: No significant geographical bias detected.")
    else:
        print("Warning: Model shows potential geographical bias.")

if __name__ == "__main__":
    run_bias_check()