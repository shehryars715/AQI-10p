# 🌍 Pearls AQI Predictor

**Real-time Air Quality Index forecasting for Pakistani cities using machine learning.**

Air pollution is one of Pakistan's most pressing environmental challenges. Lahore regularly ranks among the most polluted cities in the world, with AQI levels frequently exceeding 300 (Hazardous) during winter smog season. Karachi, Islamabad, and Peshawar also face severe air quality crises driven by vehicle emissions, industrial activity, brick kilns, and crop burning.

This project predicts the Air Quality Index (AQI) for the next 3 days across 8 major Pakistani cities. It fetches live pollutant and weather data from free APIs, trains a Random Forest model on accumulated historical data, and serves predictions through an interactive Streamlit dashboard.

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐
│   AQICN API      │     │  OpenWeather API │
│ (AQI + PM2.5,    │     │ (Temperature,    │
│  PM10, O₃, etc.) │     │  humidity, wind) │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
          ┌─────────────────┐
          │   fetcher.py    │  ← Fetches live data, merges into one row per city
          └────────┬────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  ┌─────────────┐    ┌──────────────┐
  │ aqi_data.csv│    │   app.py     │  ← Streamlit dashboard (live view)
  │ (historical │    │              │
  │  data store)│    │ Heuristic:   │
  └──────┬──────┘    │ tomorrow ≈   │
         │           │ today ± 5%   │
         ▼           └──────────────┘
  ┌─────────────┐
  │  train.py   │  ← Trains Random Forest on accumulated CSV data
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ model.joblib │  ← Saved model + scaler + feature list
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   app.py    │  ← Reloads dashboard, now with ML predictions
  │ (with model)│
  └─────────────┘
```

### Data Pipeline
1. **Fetch** — `fetcher.py` calls the free AQICN and OpenWeather APIs for 8 Pakistani cities. Each call returns current AQI, pollutant concentrations (PM2.5, PM10, O₃, NO₂, SO₂, CO), and weather readings (temperature, humidity, pressure, wind speed).
2. **Store** — Data is appended to `aqi_data.csv` with timestamps. Run this hourly to build a training dataset.
3. **Train** — `train.py` reads the CSV, engineers time features (hour, day of week, month) and lag features (AQI 1h/6h/24h ago, rolling means), then trains a Random Forest regressor that predicts AQI at +24h, +48h, and +72h simultaneously.
4. **Predict** — `app.py` loads the trained model and serves predictions. Before a model is trained, it falls back to a heuristic (tomorrow ≈ today with diurnal variation).

### Model
- **Algorithm**: Random Forest Regressor (100 trees, max depth 15)
- **Inputs**: Current AQI, PM2.5, PM10, temperature, humidity, wind speed, hour of day, day of week, month, lag features (AQI at t-1h, t-6h, t-24h), rolling means (6h, 24h)
- **Outputs**: 3 values — predicted AQI at +24 hours, +48 hours, +72 hours
- **Evaluation**: RMSE on a chronological 80/20 test split

---

## Dashboard

The Streamlit dashboard has a sidebar to select any of the 8 tracked cities and shows:

| Section | Content |
|---------|---------|
| **Current AQI** | Live AQI number with EPA category (Good → Hazardous) and color coding |
| **Pollutants** | Bar chart of PM2.5, PM10, O₃, NO₂, SO₂, CO levels |
| **3-Day Forecast** | Three cards showing predicted AQI for tomorrow, day after, and day 3 |
| **All Cities** | Comparison table of all 8 cities with color-coded AQI values |

AQI categories follow the US EPA scale:

| AQI Range | Category | Color |
|-----------|----------|-------|
| 0–50 | Good | 🟢 Green |
| 51–100 | Moderate | 🟡 Yellow |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 Orange |
| 151–200 | Unhealthy | 🔴 Red |
| 201–300 | Very Unhealthy | 🟣 Purple |
| 301+ | Hazardous | 🟤 Maroon |

---

## Project Structure

```
AQI/
├── app.py             # Streamlit dashboard
├── train.py           # Model training script
├── fetcher.py         # API client (AQICN + OpenWeather)
├── requirements.txt   # Python dependencies
├── .env.example       # API key template
├── .gitignore
├── pyproject.toml
└── README.md
```

That's it. Four source files. No Hopsworks, no Airflow, no Docker, no microservices.

---

## Setup

### 1. Get free API keys

| API | Signup Link | Free Tier |
|-----|-------------|-----------|
| **AQICN** | https://aqicn.org/data-api/token/ | 1,000 calls/day |
| **OpenWeather** | https://home.openweathermap.org/api_keys | 1,000 calls/day |

Both take under a minute to register.

### 2. Install dependencies

```bash
# Python 3.12+ required
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your keys:

```
AQICN_API_TOKEN=your_aqicn_token_here
OPENWEATHER_API_KEY=your_openweather_key_here
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. You'll see live AQI data immediately.

### 5. (Optional) Train your own model

After accumulating some data (at least 3 days of hourly readings):

```bash
# Save a snapshot of current data
python -c "from fetcher import fetch_all, save_to_csv; save_to_csv(fetch_all())"

# Train the model
python train.py
```

The dashboard automatically detects `model.joblib` and switches from heuristic to ML predictions.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Fetching | AQICN API, OpenWeather API |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Dashboard | Streamlit, Plotly |
| Config | python-dotenv |
| Model Serialization | joblib |
