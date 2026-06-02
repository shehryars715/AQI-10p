# Pearls AQI Predictor

## Setup (2 minutes)

### 1. Get free API keys
- **AQICN**: Go to https://aqicn.org/data-api/token/ — click "Get token" (free, 1000 calls/day)
- **OpenWeather**: Go to https://home.openweathermap.org/api_keys — sign up, get free key

### 2. Install & configure
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — paste your two API keys
```

### 3. Run
```bash
streamlit run app.py
```

That's it. The dashboard will open in your browser showing current AQI and 3-day predictions.

### 4. Optional: train your own model
```bash
python train.py    # trains on collected data
```
After training, the dashboard automatically uses your model instead of heuristics.

## How it works
1. **app.py** — streamlit dashboard, fetches live AQI from AQICN, predicts next 3 days
2. **train.py** — trains a RandomForest on hourly AQI data, saves to `model.joblib`
3. **fetcher.py** — API helper that gets AQI + weather data
