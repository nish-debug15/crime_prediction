# Proactive AI Crime Prediction + Prevention System

> Using machine learning to predict crime hotspots and help law enforcement act proactively — built with Python, deployed on Vercel.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-green?logo=fastapi)
![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?logo=vercel)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is this?

This project predicts **where and when crimes are likely to happen** using historical crime data and machine learning. It surfaces results through a simple web dashboard that shows heatmaps, risk scores, and patrol suggestions.

Think of it as a data science + web dev project that solves a real-world problem — great for a portfolio or college project.

---

## Features

- **Exploratory Data Analysis (EDA)** — visualize crime trends by location, time, and type
- **ML Model** — predict crime hotspots using LightGBM / Random Forest
- **Heatmap Dashboard** — interactive map showing predicted risk zones
- **Anomaly Alerts** — flag unusual spikes in crime patterns
- **Explainability** — SHAP values so predictions aren't a black box
- **Bias Check** — basic fairness audit built in

---

## Tech Stack

| Layer | Tech |
|---|---|
| Language | Python 3.10+ |
| EDA & ML | Pandas, Scikit-learn, LightGBM, SHAP, Matplotlib, Seaborn |
| Web Framework | FastAPI |
| Frontend | Jinja2 Templates + Plotly.js (served by FastAPI) |
| Database | SQLite (dev) / PlanetScale MySQL (prod) |
| Deployment | Vercel |

No separate frontend framework — everything is Python. Simple and clean.

---

## Project Structure

```
crime-prediction/
│
├── api/
│   └── index.py          # FastAPI app (Vercel entry point)
│
├── ml/
│   ├── eda.ipynb         # Exploratory Data Analysis notebook
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction logic
│   └── bias_audit.py     # Fairness checks
│
├── templates/
│   ├── index.html        # Dashboard
│   └── explain.html      # SHAP explanation view
│
├── static/
│   └── style.css
│
├── data/
│   └── LA_dataset.csv  (LA open data)
│
├── models/
│   └── hotspot_model.pkl # Saved trained model
│
├── requirements.txt
├── vercel.json
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/nish-debug15/crime_prediction.git
cd crime-prediction
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run EDA (optional but recommended)

Open the notebook to explore the dataset before training:

```bash
jupyter notebook ml/eda.ipynb
```

### 4. Train the model

```bash
python ml/train.py
```

This saves `models/hotspot_model.pkl`.

### 5. Run the app locally

```bash
uvicorn api.index:app --reload
```

Visit `http://localhost:8000` to see the dashboard.

---

## Deploying to Vercel

This project is set up for Vercel's Python serverless runtime.

```bash
npm i -g vercel   # install Vercel CLI once
vercel            # follow the prompts
```

Make sure your `vercel.json` looks like this:

```json
{
  "builds": [{ "src": "api/index.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "api/index.py" }]
}
```

That's it — Vercel handles the rest.

---

## Dataset

We have used publicly available crime data:

- [LA Crime Dataset 2020-2024](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-2024/2nrs-mtv8/about_data)

Download CSV and drop it into `data/`.

---

## ML Pipeline (Quick Overview)

```
Raw CSV → Clean & Preprocess → Feature Engineering → Train Model → Evaluate → Save Model
                                                                        ↓
                                                              SHAP Explainability
                                                                        ↓
                                                              Bias / Fairness Audit
```

**Features used:** grid cell location, hour of day, day of week, month, crime type history, weather (optional)

**Model:** LightGBM classifier → outputs crime probability per grid cell per time window

**Explainability:** SHAP waterfall plots show which features drove each prediction

---

## Ethical Considerations

This is a decision-support tool, not an automated system. A few hard rules:

- No individual profiling — predictions are for areas, not people
- Race, religion, ethnicity are never model features
- Every prediction has a SHAP explanation attached
- A basic disparate impact check runs after every model training
- Human review is required before any real-world action

---

## Requirements

```
fastapi
uvicorn
pandas
scikit-learn
lightgbm
shap
matplotlib
seaborn
plotly
jinja2
python-multipart
joblib
```

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-idea`
3. Commit and push
4. Open a pull request

---

## License

MIT — free to use, modify, and build on.

---

## Disclaimer

This project is academic and experimental. Predictions are probabilistic and should never be used as the sole basis for any law enforcement decision. Always involve qualified human judgment.
