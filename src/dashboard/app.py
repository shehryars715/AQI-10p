"""Streamlit dashboard entry point — multi-page app for AQI forecasting.

Usage:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title("🌍 Pearls AQI Predictor")
st.sidebar.markdown("---")

city = st.sidebar.selectbox(
    "📍 Select City",
    ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "🔄 Data refreshes every hour\n\n"
    "🤖 Model retrained daily\n\n"
    "⚡ Powered by Hopsworks + GitHub Actions"
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Pearls AQI Predictor")

# ── Pages ──────────────────────────────────────────────────────────
from src.dashboard.pages import (
    forecast,
    model_comparison,
    feature_importance,
    alerts,
)

pg = st.navigation(
    [
        st.Page(forecast.show, title="AQI Forecast", icon="🌤️"),
        st.Page(model_comparison.show, title="Model Comparison", icon="📊"),
        st.Page(feature_importance.show, title="Feature Importance", icon="🔬"),
        st.Page(alerts.show, title="Alerts", icon="⚠️"),
    ]
)

pg.run()
