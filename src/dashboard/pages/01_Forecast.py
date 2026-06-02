"""Page 1: AQI Forecast — current AQI gauge, 3-day prediction cards, trend chart."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

from src.dashboard.components.aqi_gauge import aqi_category_label, aqi_color, aqi_gauge
from src.dashboard.components.charts import aqi_trend_chart, pollutant_breakdown

API_BASE = "http://localhost:8000/api/v1"


def show(city: str = "Delhi") -> None:
    st.title("🌍 AQI Forecast")
    st.caption(f"Showing forecast for **{city}** — auto-refreshes every 5 minutes")

    # Fetch prediction from API
    try:
        resp = requests.get(f"{API_BASE}/predict", params={"city": city}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        st.warning("⚠️ Could not connect to the prediction API. Make sure the FastAPI server is running on port 8000.")
        _render_demo_content(city)
        return

    predictions = data.get("predictions", [])

    # ── Current AQI Gauge ──
    current_aqi = predictions[0]["aqi_predicted"] if predictions else 150
    cat = aqi_category_label(current_aqi)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(aqi_gauge(current_aqi), use_container_width=True)
    with col2:
        st.markdown(f"## Current AQI: **{current_aqi:.0f}**")
        st.markdown(
            f"### <span style='color:{aqi_color(current_aqi)}'>{cat}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {data.get('model', {}).get('type', 'unknown')}")
        st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # ── 3-Day Forecast Cards ──
    st.subheader("📅 3-Day Forecast")
    cols = st.columns(len(predictions) if predictions else 3)
    for i, pred in enumerate(predictions):
        with cols[i]:
            aqi_val = pred["aqi_predicted"]
            cat_label = pred["aqi_category"]
            color = aqi_color(aqi_val)
            st.markdown(
                f"""
                <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center;">
                    <h4>{pred['date']}</h4>
                    <h1 style="color:{color};">{aqi_val:.0f}</h1>
                    <p>{cat_label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── AQI Trend Chart ──
    st.subheader("📈 AQI Trend")
    # Build a simple trend from the 3 forecast days
    trend_df = pd.DataFrame(
        {
            "date": [pred["date"] for pred in predictions],
            "aqi_predicted": [pred["aqi_predicted"] for pred in predictions],
        }
    )
    st.plotly_chart(aqi_trend_chart(trend_df, predicted_col="aqi_predicted"), use_container_width=True)

    # ── Pollutant Breakdown ──
    st.subheader("🧪 Pollutant Levels")
    try:
        feat_resp = requests.get(f"{API_BASE}/features/latest", params={"city": city}, timeout=10)
        if feat_resp.ok:
            feat_data = feat_resp.json()
            pollutants = {
                "PM2.5": feat_data.get("pm25"),
                "PM10": feat_data.get("pm10"),
                "O₃": feat_data.get("o3"),
                "NO₂": feat_data.get("no2"),
                "SO₂": feat_data.get("so2"),
                "CO": feat_data.get("co"),
            }
            st.plotly_chart(pollutant_breakdown(pollutants), use_container_width=True)
    except Exception:
        st.info("Pollutant data will appear here when the feature pipeline runs.")


def _render_demo_content(city: str) -> None:
    """Show sample content when the API is not reachable."""
    st.info("Showing demo content — start the API server for live predictions.")
    st.plotly_chart(aqi_gauge(155), use_container_width=True)
