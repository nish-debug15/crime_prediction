# 🔍 AI Crime Prediction + Prevention System

> Predicting where and when crimes are likely to occur using advanced machine learning — with a strong ethical foundation.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)
![React Native](https://img.shields.io/badge/React_Native-0.74+-61DAFB?logo=react)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange?logo=mysql)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Ethics](https://img.shields.io/badge/Ethics-First-purple)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [ML Pipeline](#ml-pipeline)
- [Ethical Framework](#ethical-framework)
- [Database Schema](#database-schema)
- [Mobile App (React Native)](#mobile-app-react-native)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Overview

The **AI Crime Prediction + Prevention System** is an end-to-end platform that assists law enforcement agencies in proactive crime prevention. It uses spatio-temporal machine learning models, anomaly detection, and Bayesian inference to identify crime hotspots and temporal patterns — before crimes occur.

The system is designed with **transparency, fairness, and accountability** at its core. Every prediction is explainable, every model is auditable, and bias mitigation is not optional — it's built in.

> ⚠️ This system is a **decision-support tool**, not a decision-making authority. All outputs must be reviewed by qualified human officers before any operational action is taken.

---

## Key Features

| Feature | Description |
|---|---|
| 🗺️ **Crime Hotspot Heatmaps** | Interactive, temporally animated heatmaps of predicted crime zones |
| 🚨 **Real-Time Anomaly Alerts** | Multi-channel push alerts (mobile, email, webhook) for spike detections |
| 🚔 **Patrol Optimization** | ML-driven suggestions for optimal patrol routes and resource allocation |
| 🧠 **Explainable AI (XAI)** | SHAP-based explanations for every prediction — no black boxes |
| ⚖️ **Bias Detection & Mitigation** | Automated fairness audits against demographic and geographic proxies |
| 📊 **Dashboard Analytics** | Real-time command-center dashboard with drill-down analytics |
| 🔐 **Role-Based Access Control** | Granular RBAC — analysts, officers, admins, auditors |
| 📁 **Audit Logging** | Immutable audit trail of all predictions, accesses, and overrides |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Native App                  │
│         (Officers / Analysts / Commanders)          │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS / REST
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Backend                    │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Auth/RBAC  │  │ Prediction  │  │  Alert Mgr  │  │
│  │  Module    │  │   Engine    │  │   Module    │  │
│  └────────────┘  └──────┬──────┘  └─────────────┘  │
│                         │                           │
│  ┌──────────────────────▼──────────────────────┐   │
│  │              ML Pipeline Service            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │   │
│  │  │  LGBM /  │ │ Bayesian │ │   Anomaly   │ │   │
│  │  │  XGBoost │ │ Inference│ │  Detection  │ │   │
│  │  └──────────┘ └──────────┘ └─────────────┘ │   │
│  │  ┌──────────────────────────────────────┐   │   │
│  │  │       SHAP Explainability Layer      │   │   │
│  │  └──────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────┐   │   │
│  │  │     Bias Detection & Fairness Audit  │   │   │
│  │  └──────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│                    MySQL 8.0                        │
│  crime_events │ predictions │ patrol_logs │ audits  │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
- **Python 3.11+** — Core language
- **FastAPI** — Async REST API framework
- **SQLAlchemy 2.0** — ORM with async support
- **Celery + Redis** — Background tasks & job queue
- **scikit-learn, LightGBM, XGBoost** — ML models
- **SHAP** — Explainability
- **PyMC / Stan** — Bayesian inference
- **Geopandas + Shapely** — Geospatial processing
- **Pydantic v2** — Data validation

### Database
- **MySQL 8.0** — Primary relational store
- **Redis** — Caching & message broker

### Mobile (Frontend)
- **React Native 0.74+** — Cross-platform mobile (iOS + Android)
- **Expo** — Development toolchain
- **React Navigation** — Routing
- **Mapbox Maps SDK** — Interactive map rendering
- **React Query** — Server state management
- **Zustand** — Client state management

### Infrastructure
- **Docker + Docker Compose** — Containerisation
- **Nginx** — Reverse proxy
- **GitHub Actions** — CI/CD

---

## Project Structure

```
ai-crime-prediction/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── auth.py
│   │   │   │   ├── predictions.py
│   │   │   │   ├── hotspots.py
│   │   │   │   ├── alerts.py
│   │   │   │   ├── patrols.py
│   │   │   │   └── audit.py
│   │   │   └── deps.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── security.py
│   │   │   └── logging.py
│   │   ├── db/
│   │   │   ├── models.py
│   │   │   ├── session.py
│   │   │   └── migrations/
│   │   ├── ml/
│   │   │   ├── pipeline.py
│   │   │   ├── hotspot_model.py
│   │   │   ├── temporal_model.py
│   │   │   ├── anomaly_detector.py
│   │   │   ├── bayesian_inference.py
│   │   │   ├── explainability.py
│   │   │   └── bias_audit.py
│   │   ├── services/
│   │   │   ├── prediction_service.py
│   │   │   ├── alert_service.py
│   │   │   └── patrol_optimizer.py
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── mobile/
│   ├── src/
│   │   ├── screens/
│   │   │   ├── DashboardScreen.tsx
│   │   │   ├── HeatmapScreen.tsx
│   │   │   ├── AlertsScreen.tsx
│   │   │   ├── PatrolScreen.tsx
│   │   │   └── ExplainScreen.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── store/
│   │   ├── api/
│   │   └── navigation/
│   ├── app.json
│   ├── package.json
│   └── tsconfig.json
│
├── infra/
│   ├── docker-compose.yml
│   ├── nginx/
│   └── .env.example
│
├── docs/
│   ├── ETHICAL_FRAMEWORK.md
│   ├── BIAS_AUDIT_REPORT.md
│   ├── API_SPEC.yaml
│   └── DATA_DICTIONARY.md
│
├── scripts/
│   ├── seed_data.py
│   ├── train_models.py
│   └── run_bias_audit.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+ & npm
- MySQL 8.0
- Redis 7+
- Docker & Docker Compose (recommended)

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ai-crime-prediction.git
cd ai-crime-prediction
```

---

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp ../infra/.env.example .env
# Edit .env with your DB credentials and secrets

# Run DB migrations
alembic upgrade head

# Seed sample data (optional, for development)
python ../scripts/seed_data.py

# Train models (required before first run)
python ../scripts/train_models.py

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

### 3. Mobile App Setup

```bash
cd mobile

# Install dependencies
npm install

# Start Expo dev server
npx expo start
```

Scan the QR code with Expo Go (Android/iOS) or run on a simulator.

---

### 4. Docker Compose (Full Stack)

```bash
cd infra

# Copy and configure environment
cp .env.example .env
# Edit .env with your secrets

# Build and start all services
docker compose up --build

# Run migrations inside container
docker compose exec backend alembic upgrade head
```

Services will be available at:
- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Redis: `localhost:6379`
- MySQL: `localhost:3306`

---

## Environment Variables

```env
# App
APP_ENV=development
SECRET_KEY=your-secret-key-here
API_VERSION=v1

# Database
DB_HOST=localhost
DB_PORT=3306
DB_NAME=crime_prediction_db
DB_USER=root
DB_PASSWORD=yourpassword

# Redis
REDIS_URL=redis://localhost:6379/0

# ML
MODEL_PATH=./models/
RETRAIN_SCHEDULE_CRON=0 2 * * *   # 2AM daily

# Notifications
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=yourpassword
FCM_SERVER_KEY=your-firebase-cloud-messaging-key

# Ethics & Compliance
BIAS_AUDIT_ENABLED=true
BIAS_AUDIT_THRESHOLD=0.1        # Max allowed disparity
AUDIT_LOG_RETENTION_DAYS=365
```

---

## API Reference

All endpoints are prefixed with `/api/v1`. Authentication uses JWT Bearer tokens.

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/login` | Obtain access + refresh tokens |
| `POST` | `/auth/refresh` | Refresh access token |
| `POST` | `/auth/logout` | Revoke token |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/predictions/hotspots` | Get current hotspot predictions with heatmap data |
| `GET` | `/predictions/temporal` | Get temporal crime probability for a time window |
| `GET` | `/predictions/{id}/explain` | Get SHAP explanation for a specific prediction |
| `POST` | `/predictions/query` | Run a custom prediction query with filters |

### Alerts

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/alerts` | List active anomaly alerts |
| `GET` | `/alerts/{id}` | Get alert details |
| `PATCH` | `/alerts/{id}/acknowledge` | Mark alert as acknowledged |
| `POST` | `/alerts/subscribe` | Subscribe to alert channels |

### Patrol

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/patrols/optimize` | Get ML-optimized patrol route suggestions |
| `POST` | `/patrols/log` | Log patrol activity |
| `GET` | `/patrols/coverage` | Get area coverage analytics |

### Audit

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/audit/logs` | Retrieve immutable audit logs (admin only) |
| `GET` | `/audit/bias-report` | Latest bias audit report |
| `POST` | `/audit/bias-report/run` | Trigger manual bias audit |

Full OpenAPI spec available at `docs/API_SPEC.yaml`.

---

## ML Pipeline

### Models Used

**1. Spatio-Temporal Hotspot Model**
- LightGBM / XGBoost ensemble
- Features: grid cell coordinates, hour, day of week, month, weather, events, historical crime rates, demographic-neutral proxies
- Output: Crime probability score per grid cell per time window

**2. Bayesian Temporal Inference**
- PyMC-based hierarchical model
- Captures seasonality, trend, and uncertainty estimates
- Outputs credible intervals alongside point estimates

**3. Anomaly Detection**
- Isolation Forest + statistical control charts
- Flags unusual spikes in crime rates beyond baseline
- Triggers real-time alerts

**4. Patrol Optimizer**
- Constraint-based optimization (PuLP / OR-Tools)
- Inputs: hotspot predictions, officer availability, patrol constraints
- Outputs: suggested routes with expected coverage impact

### Explainability

Every prediction is accompanied by a SHAP waterfall chart showing the contribution of each feature to the output. Officers can always answer **"why was this flagged?"**

```python
# Example: generating a SHAP explanation
from ml.explainability import SHAPExplainer

explainer = SHAPExplainer(model=hotspot_model)
explanation = explainer.explain(input_features=grid_cell_features)
# Returns: feature contributions, base value, prediction value
```

### Retraining

Models are retrained nightly via a Celery scheduled task using the latest available crime data. Training metrics and bias audit results are logged and versioned.

---

## Ethical Framework

This system operates under a strict ethical policy. Full details in `docs/ETHICAL_FRAMEWORK.md`.

### Core Principles

**1. Human Oversight Is Mandatory**  
No automated action is ever taken on predictions. All outputs are advisory. A qualified officer must review and approve before any deployment decision.

**2. No Individual Profiling**  
The system predicts for geographic areas and time windows — never for individuals. Personal data is not processed or stored.

**3. Protected Attribute Exclusion**  
Race, ethnicity, religion, gender, and immigration status are **never used** as model features — directly or as proxies.

**4. Transparency by Default**  
Every prediction includes a human-readable SHAP explanation. Black-box outputs are not permitted.

**5. Bias Auditing**  
Automated fairness audits run on every model version before deployment. Disparate impact is measured across geographic and demographic dimensions. Models exceeding the bias threshold are blocked from deployment.

**6. Right to Contestation**  
Affected parties have the right to contest predictions. An independent audit process is available.

**7. Data Minimisation**  
Only the minimum necessary data is collected. Personal identifiers are never stored. Data is retained only as long as legally required.

**8. Accountability**  
Every prediction access, override, and action is recorded in an immutable audit log accessible to oversight bodies.

---

## Mobile App (React Native)

### Screens

| Screen | Description |
|--------|-------------|
| **Dashboard** | Live summary — active alerts, hotspot count, patrol coverage % |
| **Heatmap** | Animated spatio-temporal heatmap with time scrubber |
| **Alerts** | Real-time anomaly alerts with acknowledge action |
| **Patrol** | Patrol route suggestions with map overlay |
| **Explain** | SHAP explanation viewer for any prediction |
| **Audit Log** | Audit trail viewer (admin/auditor role only) |

### Push Notifications

Alerts are delivered via Firebase Cloud Messaging (FCM). Officers subscribe to specific grid zones. Notifications include severity, location, and a deep-link to the relevant alert screen.

---

## Contributing

We welcome contributions! Please read `CONTRIBUTING.md` first.

### Guidelines

1. Fork the repo and create your branch: `git checkout -b feature/your-feature`
2. All ML changes must include a bias audit result in the PR
3. Write tests for new API endpoints (pytest)
4. Follow PEP 8 for Python, ESLint + Prettier for TypeScript
5. Open a PR with a clear description of changes and ethical implications (if any)

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Disclaimer

> This system is intended solely as a **decision-support tool** for trained law enforcement professionals. Predictions are probabilistic in nature and should never be the sole basis for law enforcement action. The developers disclaim all liability for misuse. Deployment must comply with all applicable local, national, and international laws, including data protection regulations (e.g. GDPR, PDPA, CCPA). An independent ethics review is strongly recommended before any real-world deployment.

---

<div align="center">
  Built with intention. Deployed with caution. Audited always.
</div>
