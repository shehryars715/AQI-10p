"""Page 4: Alerts — Hazardous AQI alerts, city comparison, historical trends."""

from __future__ import annotations

from datetime import date, timedelta

import requests
import streamlit as st

from src.dashboard.components.alerts import city_comparison_table, hazardous_alert_card

API_BASE = "http://localhost:8000/api/v1"


def show(city: str = "Delhi") -> None:
    st.title("⚠️ AQI Alerts & Hazardous Days")

    # ── Fetch batch predictions for all cities ──
    try:
        resp = requests.get(f"{API_BASE}/predict/batch", timeout=10)
        resp.raise_for_status()
        all_preds = resp.json().get("predictions", {})
    except Exception:
        st.warning("⚠️ Could not connect to the API.")
        all_preds = {}

    # ── Active Alerts ──
    st.subheader("🚨 Active Alerts")
    has_alerts = False
    for city_name, preds in all_preds.items():
        for pred in preds:
            if pred.get("aqi_category") in ("Unhealthy", "Very Unhealthy", "Hazardous"):
                has_alerts = True
                hazardous_alert_card(
                    day=pred["date"],
                    aqi=pred["aqi_predicted"],
                    city=city_name,
                    category=pred["aqi_category"],
                )

    if not has_alerts:
        st.success("✅ No hazardous AQI days predicted for the next 3 days.")

    st.divider()

    # ── City Comparison Table ──
    st.subheader("🏙️ City Comparison (Next 3 Days)")
    if all_preds:
        city_comparison_table(all_preds)
    else:
        st.info("No prediction data available.")

    st.divider()

    # ── AQI Category Reference ──
    st.subheader("📋 AQI Category Reference")
    _render_aqi_reference()


def _render_aqi_reference() -> None:
    """Render the EPA AQI category reference table."""
    categories = [
        ("🟢", "Good", "0–50", "Air quality is satisfactory."),
        ("🟡", "Moderate", "51–100", "Acceptable; moderate concern for sensitive individuals."),
        ("🟠", "USG", "101–150", "Sensitive groups may experience health effects."),
        ("🔴", "Unhealthy", "151–200", "Everyone may experience health effects."),
        ("🟣", "Very Unhealthy", "201–300", "Health alert: serious health effects for all."),
        ("🟤", "Hazardous", "301+", "Health warning: emergency conditions."),
    ]
    for emoji, cat, aqi_range, desc in categories:
        st.caption(f"{emoji} **{cat}** ({aqi_range}): {desc}")
