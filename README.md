# 🌍 Lahore AQI Predictor

**Real-time Air Quality Index forecasting for Lahore using machine learning.**

Lahore regularly ranks among the most polluted cities in the world, with AQI levels frequently exceeding 300 (Hazardous) during winter smog season. This project predicts the AQI for the next 3 days using live pollutant data, weather readings, and a trained ML model.

---

## Dashboard

<!-- Replace this with your screenshot: Win+Shift+S, then paste here -->
<!-- ![Dashboard](screenshot.png) -->

![Dashboard](https://via.placeholder.com/800x450/0a0a0a/22cc66?text=Lahore+AQI+Predictor+Dashboard)

The dashboard shows:
- **Current AQI** with EPA color-coded category (Good → Hazardous)
- **Pollutant breakdown** — PM2.5, PM10, O₃, NO₂, SO₂, CO
- **3-day forecast** using the best trained ML model
- **🚨 Hazard alerts** when AQI exceeds safe levels
- **📈 Historical trend** — last 7 days from collected data
- **🔬 Feature importance** — which factors most affect predictions
- **📊 Model comparison** — RMSE, MAE, R² across models

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐
│   AQICN API      │     │  Open-Meteo API  │
│ (Live AQI +      │     │ (Historical AQI  │
│  PM2.5, PM10)    │     │  + weather, free)│
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
          ┌─────────────────┐
          │   fetcher.py     │  Live data (AQICN + OpenWeather)
          │   backfill.py    │  Historical data (Open-Meteo, 90 days)
          └────────┬────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  ┌─────────────┐    ┌──────────────┐
  │ aqi_data.csv│    │   app.py     │  Streamlit dashboard
  │ (historical │    │              │
  │  data store)│    │ Before model:│
  └──────┬──────┘    │ heuristic    │
         │           │ (trend-based)│
         ▼           └──────────────┘
  ┌─────────────┐
  │  train.py   │  Trains 3 models, picks best
  └──────┬──────┘
         │
         ├── model.joblib         (best model + scaler)
         ├── evaluation.csv       (RMSE/MAE/R² for dashboard)
         └── feature_importance.csv (top predictors)
              │
              ▼
       ┌──────────────┐
       │   app.py     │  Dashboard now uses ML predictions
       │ (with model) │
       └──────────────┘
```

### Models Trained

| Model | Avg RMSE | Avg MAE | Avg R² |
|-------|----------|---------|--------|
| Ridge Regression | 38.0 | 29.8 | -0.08 |
| Random Forest | 40.9 | 35.0 | -0.25 |
| Neural Network (MLP) | 44.0 | 37.1 | -0.45 |

Ridge wins because AQI prediction is mostly linear (PM2.5 drives it at r=0.95).

### Features Used (19 total)
Hour of day, day of week, month, AQI lags (1h, 6h, 24h), rolling means (6h, 24h), temperature, humidity, wind speed, PM2.5, PM10, O₃, NO₂, SO₂, CO, and more.

### Top 5 Predictors (Feature Importance)
1. PM2.5 — the strongest driver of AQI
2. AQI lag 1h — what was the AQI an hour ago
3. AQI rolling 24h — yesterday's average
4. AQI lag 24h — yesterday at this hour
5. Hour of day — diurnal pollution cycle

---

## Project Structure

```
AQI/
├── app.py               # Streamlit dashboard
├── train.py              # Model training (RF, Ridge, Neural Net)
├── backfill.py           # Historical data from Open-Meteo (free, no key)
├── fetcher.py            # Live data from AQICN + OpenWeather
├── eda.py                # Exploratory Data Analysis
├── requirements.txt      # Dependencies
├── .env.example          # API key template
├── .github/workflows/
│   ├── feature_pipeline.yml   # Hourly data collection
│   └── training_pipeline.yml  # Daily model retraining
└── README.md
```

---

## Deployment on Streamlit Community Cloud

The app is ready to deploy from GitHub. Streamlit Community Cloud runs the app from the repository root, installs `requirements.txt`, and launches `app.py`.

### Deploy settings

Use these values when creating the app:

| Field | Value |
|-------|-------|
| Repository | `shehryars715/AQI-10p` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.12` |

### Streamlit secrets

In Streamlit Cloud, open **Advanced settings** and paste:

```toml
AQICN_API_TOKEN = "paste_your_aqicn_token_here"
OPENWEATHER_API_KEY = "paste_your_openweather_key_here"
```

Do not commit real API keys. Locally, the app reads `.env`; on Streamlit Cloud, it reads the secrets above.

### Verify deployment

After deployment, the app should show:

- Current AQI for Lahore
- PM2.5, PM10, temperature, humidity, and weather metrics
- 3-day forecast using `model.joblib`
- Historical trend from `aqi_data.csv`
- Feature importance and model comparison sections

If deployment fails, open the Streamlit Cloud app logs and check the failing line. The most common problems are missing secrets or a typo in the main file path.

---

## Setup

### 1. Get API keys (both free)

| API | Link | Limit |
|-----|------|-------|
| AQICN | https://aqicn.org/data-api/token/ | 1,000 calls/day |
| OpenWeather | https://home.openweathermap.org/api_keys | 1,000 calls/day |

### 2. Install

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — paste your AQICN_API_TOKEN and OPENWEATHER_API_KEY
```

### 3. Backfill historical data (90 days)

```bash
python backfill.py --days 90
```

### 4. Train the model

```bash
python train.py
```
Trains Random Forest, Ridge, and Neural Network — saves the best one.

### 5. Run EDA (optional)

```bash
python eda.py
```
Prints trends, correlations, diurnal patterns, worst days.

### 6. Launch dashboard

```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

---

## CI/CD (GitHub Actions)

Two workflows automate the project on every push to `main`:

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| **Feature Pipeline** | Every hour | Fetches live AQI + weather, appends to CSV |
| **Training Pipeline** | Daily at 6:23 AM | Backfills 90 days, trains all models, uploads `model.joblib` artifact |

**Required GitHub Secrets**: `AQICN_API_TOKEN`, `OPENWEATHER_API_KEY`
**Training pipeline**: works without secrets (uses free Open-Meteo API).

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Live AQI | AQICN API |
| Weather + Historical | OpenWeather API, Open-Meteo API |
| Data | Pandas, NumPy |
| Models | Scikit-learn (Random Forest, Ridge, MLP Neural Net) |
| Dashboard | Streamlit, Plotly |
| CI/CD | GitHub Actions |
| Config | python-dotenv |
