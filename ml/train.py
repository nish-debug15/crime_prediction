import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import shap
import joblib
import matplotlib.pyplot as plt
import os

print("Starting ML Pipeline: Hotspot Classification")

try:
    df = pd.read_csv("data/cleaned_crime_data.csv")
    print(f"Data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: data/LA_dataset.csv not found.")
    exit()

df = df.dropna(subset=['lat_grid', 'lon_grid', 'hour', 'day_of_week', 'month', 'is_weekend'])
df = df[(df['lat_grid'] != 0) & (df['lon_grid'] != 0)]

df['crime_count'] = df.groupby(['lat_grid', 'lon_grid', 'hour'])['dr_no'].transform('count')
threshold = df['crime_count'].quantile(0.75)
df['is_hotspot'] = (df['crime_count'] > threshold).astype(int)
print(f"Hotspot threshold: > {threshold:.2f}. Hotspots: {df['is_hotspot'].sum()}")

features = ['lat_grid', 'lon_grid', 'hour', 'day_of_week', 'month', 'is_weekend']
X = df[features]
y = df['is_hotspot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

print("\nTraining LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/hotspot_model.pkl")
print("Model saved to models/hotspot_model.pkl")

df[['lat_grid', 'lon_grid', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_hotspot']].to_csv("data/model_features.csv", index=False)
print("model_features.csv updated with 6 features.")

print("\nGenerating SHAP plots...")
explainer = shap.TreeExplainer(model)
shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
shap_values = explainer.shap_values(shap_sample)

if isinstance(shap_values, list):
    shap_to_plot = shap_values[1]
else:
    shap_to_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

os.makedirs("static", exist_ok=True)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_to_plot, shap_sample, show=False)
plt.title("SHAP Feature Importance (High Risk Drivers)")
plt.tight_layout()
plt.savefig("static/shap_summary.png", dpi=300)
plt.close()
print("SHAP saved to static/shap_summary.png")

print("\nML Pipeline Complete!")