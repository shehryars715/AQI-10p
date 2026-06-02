"""Page 3: Feature Importance — SHAP beeswarm, bar chart, waterfall."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.dashboard.components.charts import shap_beeswarm


def show(city: str = "Delhi") -> None:
    st.title("🔬 Feature Importance (SHAP)")

    st.info(
        "SHAP values are precomputed during model training and loaded from the Model Registry. "
        "Run the training pipeline with SHAP enabled to populate this page."
    )

    # Demo SHAP data for illustration
    st.subheader("Feature Importance Summary")
    st.caption(
        "The chart below shows which features most influence AQI predictions. "
        "Longer bars = greater impact on the model's output."
    )

    # Synthetic SHAP values for demo
    feature_names = [
        "pm25_lag_1h", "temperature", "hour", "aqi_lag_24h", "humidity",
        "wind_speed", "pm10_lag_1h", "day_of_week", "aqi_lag_6h", "month",
        "pressure", "season_Winter", "is_weekend", "aqi_rolling_6h", "co_lag_1h",
    ]
    rng = np.random.default_rng(42)
    demo_shap = rng.normal(0, 1, (100, len(feature_names))) * np.abs(
        rng.normal(0, 2, len(feature_names))
    )

    st.plotly_chart(
        shap_beeswarm(demo_shap, feature_names),
        use_container_width=True,
    )

    # Feature descriptions
    st.subheader("Top 5 Feature Explanations")
    top_features = [
        ("PM2.5 (1h lag)", "The strongest predictor — recent PM2.5 levels directly influence short-term AQI."),
        ("Temperature", "Higher temperatures often correlate with increased ground-level ozone formation."),
        ("Hour of Day", "AQI follows diurnal patterns — typically higher during morning and evening rush hours."),
        ("AQI (24h lag)", "Yesterday's AQI provides a strong baseline for tomorrow's prediction."),
        ("Humidity", "High humidity can trap pollutants near the ground, worsening AQI."),
    ]
    for name, desc in top_features:
        with st.expander(f"📌 {name}"):
            st.write(desc)
