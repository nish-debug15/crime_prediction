# AI Crime Prediction + Prevention System

Predicting where and when crimes are likely to occur using advanced machine learning, with a strong ethical foundation.

## Overview

This system predicts crime hotspots and temporal patterns to assist law enforcement in proactive crime prevention. It uses spatio-temporal modeling, anomaly detection, and Bayesian inference while maintaining strict ethical standards.

### Key Features

- Crime hotspot heatmaps with temporal animation
- Real-time anomaly alerts with multi-channel notifications
- Patrol optimization suggestions using ML
- Explainable predictions with SHAP integration
- Built-in bias detection and mitigation

## Team Structure

| Role | Responsibilities |
|------|------------------|
| Nishit Patel | Model development, training, optimization |
| Pranav Adhikari | API development, data pipeline, infrastructure |
| Unique Bhakta Shrestha | Dashboard, visualization, user interface |
| Pragun Shrestha | Bias mitigation, data quality, compliance |

## Architecture

The system follows a microservices architecture with:

- Client Layer: React web and mobile applications
- API Gateway: FastAPI with authentication and rate limiting
- Microservices: Prediction, Alert, Analytics, Ethics, Patrol, Data Ingestion
- Data Layer: PostgreSQL, TimescaleDB, Redis, MongoDB, Elasticsearch
- ML Layer: Spatio-temporal models, anomaly detection, Bayesian inference

## ML Stack

### Spatio-Temporal Models
- Graph Neural Networks for spatial dependencies
- Transformer encoders for temporal patterns
- Multi-step forecasting (1-hour to 7-day predictions)

### Anomaly Detection
- Isolation Forest for general anomaly detection
- Autoencoder for pattern deviation
- DBSCAN for spatial clustering
- LSTM Autoencoder for temporal anomalies

### Bayesian Models
- Uncertainty quantification for predictions
- Confidence intervals for risk scores
- Prior knowledge incorporation

## Tech Stack

### Backend
- Python 3.9+, FastAPI, PostgreSQL, TimescaleDB, Redis, Celery, MongoDB, Elasticsearch

### Machine Learning
- PyTorch, PyTorch Geometric, scikit-learn, XGBoost, Pyro, SHAP, Fairlearn

### Frontend
- React 18, TypeScript, Mapbox GL, D3.js, Redux Toolkit, Tailwind CSS

### Infrastructure
- Vercel (deployment), GitHub Actions (CI/CD), Prometheus, Grafana

## Project Timeline

### Phase 1: Foundation (Weeks 1-4)
- Project setup and data collection
- ETL pipeline development
- Feature engineering framework

### Phase 2: Core Development (Weeks 5-12)
- ML model development
- Backend API services
- Frontend dashboard
- Service integration

### Phase 3: Ethics and Validation (Weeks 13-16)
- Bias detection implementation
- Explainability dashboard
- Testing and validation

### Phase 4: Deployment (Weeks 17-20)
- Vercel deployment configuration
- Production setup
- Launch and monitoring

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+
- PostgreSQL 14+
- Redis 6.2+

### Installation

1. Clone the repository
```
git clone https://github.com/nish-debug15/crime_prediction.git
cd crime-prediction
```

2. Set up Python environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Configure environment variables
```
cp .env.example .env
```

4. Start the backend
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. Start the frontend
```
cd frontend
npm install
npm run dev
```

### Access Points
- API Documentation: http://localhost:8000/docs
- Dashboard: http://localhost:3000

## Data Sources

- Police records (historical crime data)
- Weather API (meteorological data)
- Census data (demographics)
- OpenStreetMap (geographic data)
- Event calendars (public events)

## API Endpoints

### Predictions
- GET /api/v1/predictions/area - Crime prediction for area
- GET /api/v1/predictions/explain - Prediction with explanation
- POST /api/v1/predictions/batch - Batch predictions

### Alerts
- GET /api/v1/alerts/active - Active alerts
- POST /api/v1/alerts/rules - Configure alert rules

### Heatmaps
- GET /api/v1/heatmap - Current heatmap data
- GET /api/v1/heatmap/history - Historical heatmap

### Patrol Optimization
- GET /api/v1/patrol/recommendations - Patrol recommendations
- POST /api/v1/patrol/optimize - Optimize patrol routes

## Ethical Framework

### Bias Detection
- Demographic parity analysis
- Equal opportunity metrics
- Disparate impact testing
- Automated mitigation strategies

### Privacy Protection
- k-anonymity (k >= 5)
- Differential privacy
- Geographic aggregation
- Data retention policies

### Transparency
- Explainable predictions
- Audit logging
- Human oversight
- Regular fairness audits

## Testing

Run tests with:
```
pytest tests/ -v --cov=app
```

Test coverage targets:
- Backend API: 90%
- ML Models: 85%
- Frontend: 80%

## Deployment

The application is deployed on Vercel with:
- Automatic deployments from main branch
- Preview deployments for pull requests
- Environment variable management
- Custom domain configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Submit a pull request

All PRs require 2 approvals and must pass CI/CD checks.

## License

MIT License - see LICENSE file for details.

## Disclaimer

This system assists law enforcement in crime prevention and resource allocation. It should not be the sole basis for law enforcement decisions. All predictions must be reviewed by trained professionals and used in compliance with local laws and regulations.
