# Proactive AI Crime Prediction + Prevention System

> Decoupled microservices using ML to predict spatial-temporal crime hotspots. FastAPI backend, Streamlit dashboard, SQLite audit log.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What Is This?

Predicts **where and when crimes are likely to happen** using historical data + ML. Decoupled microservices ensure high performance, secure logging, and an interactive analytical dashboard with AI explainability (SHAP).

---

## Key Features

- **Microservices Architecture** — ML backend (FastAPI) and UI (Streamlit) fully separated
- **ML Predictive Engine** — LightGBM classifier on spatial-temporal features
- **Audit Logging** — Every API prediction logged to SQLite (`predictions.db`)
- **Interactive Dashboard** — Real-time maps and risk scoring via Streamlit
- **Explainable AI (XAI)** — SHAP values; model is not a black box
- **Algorithmic Fairness** — Disparate Impact auditing for geographic reporting bias

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Science & ML | Pandas, Scikit-learn (v1.8.0), LightGBM, SHAP |
| Backend API | FastAPI, Uvicorn |
| Frontend UI | Streamlit, Requests |
| Database | SQLite3 |

---

## Project Structure

```text
crime_prediction/
│
├── api/
│   └── index.py           # REST API (FastAPI)
│
├── data/
│   ├── LA_dataset.csv     # Raw spatial-temporal dataset
│   └── predictions.db     # Auto-generated SQLite audit DB
│
├── ml/
│   ├── eda.ipynb          # EDA & Feature Engineering
│   ├── train.py           # Model training + SHAP generation
│   ├── predict.py         # Prediction logic + DB logging
│   └── bias_audit.py      # Disparate Impact fairness checker
│
├── models/
│   └── hotspot_model.pkl  # Compiled LightGBM binary
│
├── static/
│   └── shap_summary.png   # Auto-generated explainability chart
│
├── app.py                 # Streamlit dashboard
└── requirements.txt
```

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/nish-debug15/crime_prediction.git
cd crime_prediction
```

### 2. Set Up Environment

> Use exact scikit-learn version to avoid unpickling errors.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run ML Pipeline (optional — retrain / regenerate SHAP)

```bash
python ml/train.py
python ml/bias_audit.py
```

### 4. Boot Microservices

Run backend and frontend on separate terminals.

**Terminal 1 — Backend (API + DB):**
```bash
uvicorn api.index:app --reload
# → http://127.0.0.1:8000
```

**Terminal 2 — Frontend (Streamlit):**
```bash
streamlit run app.py
# → http://localhost:8501
```

---

## Dataset

Trained on publicly available municipal crime data:

- [LA Crime Dataset 2020–2024](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-2024/2nrs-mtv8/about_data)

---

## Ethical Considerations

Decision-support tool only. Never replaces human judgment.

- **No Individual Profiling** — Predictions target geographic grids, not individuals
- **Feature Restrictions** — Race, religion, ethnicity excluded from training data
- **Transparency** — Every prediction backed by SHAP explainability
- **Accountability** — SQLite audit trail of every algorithmic decision

---

## Contributors

Nishit Patel, Pranav Adhikari, Pragun, Unique Bhakta Shrestha, Sameera Simha J

---

## License

MIT — Free to use, modify, build upon.

> **Disclaimer:** Academic and experimental. Predictions are probabilistic and must never be sole basis for any law enforcement decision.
