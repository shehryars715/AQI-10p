"""Page 2: Model Comparison — metrics table, RMSE bars, scatter plots."""

from __future__ import annotations

import requests
import streamlit as st

from src.dashboard.components.charts import model_comparison_chart

API_BASE = "http://localhost:8000/api/v1"


def show(city: str = "Delhi") -> None:
    st.title("📊 Model Comparison")

    try:
        resp = requests.get(f"{API_BASE}/models/info", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        st.warning("⚠️ Could not connect to the API.")
        _render_demo()
        return

    models = data.get("models", {})

    if not models:
        st.info("No models have been trained yet. Run the training pipeline first.")
        return

    # Metrics table
    st.subheader("Model Performance Metrics")
    _render_metrics_table(models)

    # RMSE comparison chart
    st.subheader("RMSE Comparison")
    try:
        st.plotly_chart(model_comparison_chart(models), use_container_width=True)
    except Exception:
        st.warning("Could not render comparison chart.")

    # Production model badge
    prod_model = data.get("production_model", "—")
    st.metric(label="🏆 Production Model", value=prod_model)


def _render_metrics_table(models: dict) -> None:
    """Render an HTML metrics table."""
    html = """
    <table style="width:100%; border-collapse: collapse;">
        <tr style="background-color: #f0f0f0;">
            <th>Model</th>
            <th>Avg RMSE</th>
            <th>Avg MAE</th>
            <th>Avg R²</th>
            <th>Day 1 RMSE</th>
            <th>Day 2 RMSE</th>
            <th>Day 3 RMSE</th>
        </tr>
    """
    for name, metrics in models.items():
        html += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{metrics.get('avg_rmse', '—'):.2f}</td>
            <td>{metrics.get('avg_mae', '—'):.2f}</td>
            <td>{metrics.get('avg_r2', '—'):.4f}</td>
            <td>{metrics.get('rmse_day1', '—'):.2f}</td>
            <td>{metrics.get('rmse_day2', '—'):.2f}</td>
            <td>{metrics.get('rmse_day3', '—'):.2f}</td>
        </tr>
        """
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def _render_demo() -> None:
    """Show demo model comparison."""
    demo_metrics = {
        "RandomForest": {
            "avg_rmse": 18.2, "avg_mae": 12.1, "avg_r2": 0.91,
            "rmse_day1": 15.0, "rmse_day2": 18.5, "rmse_day3": 21.0,
        },
        "Ridge": {
            "avg_rmse": 22.4, "avg_mae": 16.8, "avg_r2": 0.85,
            "rmse_day1": 19.0, "rmse_day2": 23.0, "rmse_day3": 25.2,
        },
    }
    _render_metrics_table(demo_metrics)
    st.plotly_chart(model_comparison_chart(demo_metrics), use_container_width=True)
