"""Pearls AQI Predictor — Dashboard

Usage: streamlit run app.py

A simple, single-file dashboard that shows:
- Current AQI for 5 Indian cities
- 3-day AQI forecast (model-based if trained, heuristic fallback otherwise)
- Trend chart
"""

import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from fetcher import fetch_all, fetch_aqi, fetch_weather, CITIES

load_dotenv()

# ── Page setup ──
st.set_page_config(page_title="AQI Predictor", page_icon="🌍", layout="wide")
st.title("🌍 Pearls AQI Predictor")
st.caption("Live AQI + 3-day forecast for Indian cities")

# ── Load model ──
MODEL_FILE = "model.joblib"
model_bundle = None
if os.path.exists(MODEL_FILE):
    model_bundle = joblib.load(MODEL_FILE)
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.info("ℹ️ No model yet — using trend-based prediction")

city = st.sidebar.selectbox("Select City", CITIES)

# ── AQI helpers ──
def aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def aqi_color(aqi: float) -> str:
    if aqi <= 50: return "green"
    elif aqi <= 100: return "yellow"
    elif aqi <= 150: return "orange"
    elif aqi <= 200: return "red"
    elif aqi <= 300: return "purple"
    return "maroon"

# ── Fetch data ──
@st.cache_data(ttl=300)  # cache for 5 minutes
def get_data():
    return fetch_all()

data = get_data()

if data.empty:
    st.error("Could not fetch AQI data. Check your API keys in .env")
    st.stop()

# ── Current AQI ──
city_data = data[data["city"] == city]
if city_data.empty:
    st.warning(f"No data for {city}")
    st.stop()

row = city_data.iloc[0]
current_aqi = row["aqi"]

col1, col2, col3 = st.columns(3)
col1.metric("Current AQI", f"{current_aqi:.0f}", delta=None,
            help=aqi_category(current_aqi))
col2.metric("PM2.5", f"{row.get('pm25', 'N/A')} µg/m³" if pd.notna(row.get('pm25')) else "N/A")
col3.metric("Temperature", f"{row.get('temperature', 'N/A')} °C" if pd.notna(row.get('temperature')) else "N/A")

# Category badge
st.markdown(
    f"<h3 style='color:{aqi_color(current_aqi)};'>{aqi_category(current_aqi)}</h3>",
    unsafe_allow_html=True,
)

# ── Pollutant bars ──
st.subheader("Pollutants")
pollutants = {
    "PM2.5": row.get("pm25"),
    "PM10": row.get("pm10"),
    "O₃": row.get("o3"),
    "NO₂": row.get("no2"),
    "SO₂": row.get("so2"),
    "CO": row.get("co"),
}
present = {k: v for k, v in pollutants.items() if pd.notna(v)}
if present:
    st.bar_chart(pd.Series(present), horizontal=True)
else:
    st.caption("No pollutant data available")

# ── 3-Day Prediction ──
st.subheader("📅 3-Day Forecast")

def predict_heuristic(current_aqi: float, hour: int) -> list:
    """Simple prediction: tomorrow ≈ today with diurnal variation."""
    base = current_aqi
    predictions = []
    for i, h in enumerate([24, 48, 72]):
        # Add slight trend + noise based on time of day
        hour_of_day = (hour + h) % 24
        diurnal = 5 * np.sin(hour_of_day / 24 * 2 * np.pi)  # ±5 AQI diurnal swing
        aqi_val = base + diurnal + i * 2  # slight upward drift
        predictions.append(max(0, aqi_val))
    return predictions

def predict_model(current_row) -> list:
    """Use trained model to predict."""
    if model_bundle is None:
        return None

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]

    # Build feature vector (fill missing with 0)
    X = pd.DataFrame([current_row])[features].fillna(0)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)[0]
    return preds.tolist()

current_hour = datetime.now().hour
predictions = predict_model(row.to_dict()) if model_bundle else None
if predictions is None:
    predictions = predict_heuristic(current_aqi, current_hour)

days = [(datetime.now() + timedelta(hours=h)) for h in [24, 48, 72]]
cols = st.columns(3)
for i, (day, pred) in enumerate(zip(days, predictions)):
    cat = aqi_category(pred)
    with cols[i]:
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; border:2px solid {aqi_color(pred)}; text-align:center;">
                <p>{day.strftime('%b %d')}</p>
                <h2 style="color:{aqi_color(pred)};">{pred:.0f}</h2>
                <p>{cat}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Multi-city overview ──
st.subheader("🏙️ All Cities Overview")
overview = data[["city", "aqi", "pm25", "temperature", "humidity"]].copy()
overview["Category"] = overview["aqi"].apply(aqi_category)
st.dataframe(
    overview,
    use_container_width=True,
    hide_index=True,
)

# ── Footer ──
st.divider()
st.caption(f"Data fetched at {datetime.now().strftime('%H:%M')} • "
           f"Model: {'Trained RF' if model_bundle else 'Trend-based'} • "
           f"Refresh: auto every 5 min")
