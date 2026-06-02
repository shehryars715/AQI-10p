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
st.set_page_config(page_title="Lahore AQI Predictor", page_icon="🌍", layout="wide")
st.title("🌍 Lahore AQI Predictor")
st.caption("Live AQI + 3-day forecast for Lahore")

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

def predict_model(current_row: dict) -> list:
    """Use trained model to predict, computing lag features from CSV history."""
    if model_bundle is None:
        return None

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]

    # Load history to compute lag/rolling features properly
    history = None
    if os.path.exists("aqi_data.csv"):
        history = pd.read_csv("aqi_data.csv", parse_dates=["timestamp"])
        # Keep only last 72 rows for lag computation
        history = history.sort_values("timestamp").tail(72)

    # Current row as DataFrame
    now_df = pd.DataFrame([current_row])
    if "timestamp" not in now_df.columns:
        now_df["timestamp"] = pd.Timestamp.now()

    # Combine history + current
    if history is not None and not history.empty:
        combined = pd.concat([history, now_df], ignore_index=True)
    else:
        combined = now_df

    # Apply same feature engineering as train.py
    from train import create_features
    combined = create_features(combined)

    # Get the last row (our current data with proper lags)
    last_row = combined.iloc[-1:]

    # Build feature vector
    X = pd.DataFrame(last_row)
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features].fillna(0)
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

# ── Hazardous Alert Banner ──
if current_aqi > 150:
    st.error(f"🚨 **HAZARDOUS AQI ALERT**: Current AQI is {current_aqi:.0f} ({aqi_category(current_aqi)}). "
             f"Avoid outdoor activities. Close windows. Wear N95 masks if going outside.")
elif current_aqi > 100:
    st.warning(f"⚠️ **AIR QUALITY WARNING**: Current AQI is {current_aqi:.0f} ({aqi_category(current_aqi)}). "
               f"Sensitive groups should limit outdoor exposure.")

# Check forecast for hazardous days
hazardous_days = [(day, pred) for day, pred in zip(days, predictions) if pred > 150]
if hazardous_days:
    for day, pred in hazardous_days:
        st.warning(f"📅 **{day.strftime('%b %d')}**: Predicted AQI {pred:.0f} — {aqi_category(pred)}")

# ── Analytics Sections (collapsible) ──

# Historical trend
with st.expander("📈 Historical AQI Trend (from CSV)", expanded=False):
    if os.path.exists("aqi_data.csv"):
        hist = pd.read_csv("aqi_data.csv", parse_dates=["timestamp"])
        hist = hist[hist["city"] == city].sort_values("timestamp").tail(168)  # last 7 days
        if not hist.empty:
            st.line_chart(hist.set_index("timestamp")["aqi"], height=300)
        else:
            st.caption("No historical data for this city yet.")
    else:
        st.caption("No aqi_data.csv found. Run backfill.py or the feature pipeline first.")

# Feature importance
with st.expander("🔬 Feature Importance", expanded=False):
    if os.path.exists("feature_importance.csv"):
        fi = pd.read_csv("feature_importance.csv").head(10)
        st.bar_chart(fi.set_index("feature")["importance"], horizontal=True)
        st.caption("Top 10 features by Random Forest importance (mean decrease in impurity)")
    elif model_bundle and hasattr(model_bundle["model"], "feature_importances_"):
        fi = pd.DataFrame({
            "feature": model_bundle["features"],
            "importance": model_bundle["model"].feature_importances_,
        }).sort_values("importance", ascending=False).head(10)
        st.bar_chart(fi.set_index("feature")["importance"], horizontal=True)
    else:
        st.caption("Run `python train.py` to generate feature importance.")

# Model comparison
with st.expander("📊 Model Comparison (RMSE / MAE / R²)", expanded=False):
    if os.path.exists("evaluation.csv"):
        ev = pd.read_csv("evaluation.csv")
        st.dataframe(ev, use_container_width=True, hide_index=True)
    else:
        st.caption("Run `python train.py` to generate model metrics.")

# ── Footer ──
st.divider()
st.caption(f"Data fetched at {datetime.now().strftime('%H:%M')} • "
           f"Model: {model_bundle['model_name'] if model_bundle else 'Trend-based'} • "
           f"Refresh: auto every 5 min")
