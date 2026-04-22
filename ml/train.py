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
    df = pd.read_csv("data/model_features.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: data/model_features.csv not found. Ensure EDA pipeline was run.")
    exit()

threshold = df['crime_count'].quantile(0.75)
df['is_hotspot'] = (df['crime_count'] > threshold).astype(int)
print(f"Hotspot Threshold: > {threshold:.2f} crimes. Found {df['is_hotspot'].sum()} hotspots.")

features = ['lat_grid', 'lon_grid', 'hour']
X = df[features]
y = df['is_hotspot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} samples, Evaluating on {len(X_test)} samples.")

print("\nTraining LightGBM Classifier...")
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

os.makedirs("models", exist_ok=True)
model_path = "models/hotspot_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved successfully to {model_path}")

print("\nGenerating SHAP Explainability plots...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_to_plot = shap_values[1]
else:
    shap_to_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

os.makedirs("static", exist_ok=True)
plt.figure(figsize=(10, 6))

shap.summary_plot(shap_to_plot, X_test, show=False)

plt.title("SHAP Feature Importance (High Risk Drivers)")
plt.tight_layout()
plt.savefig("static/shap_summary.png", dpi=300)
print("SHAP summary plot saved to static/shap_summary.png")

print("ML Pipeline Complete!")