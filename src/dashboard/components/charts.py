"""Reusable Plotly chart builders for the dashboard."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def aqi_trend_chart(
    df: pd.DataFrame,
    date_col: str = "date",
    actual_col: str = "aqi",
    predicted_col: Optional[str] = "aqi_predicted",
    title: str = "AQI Trend (Last 7 Days)",
) -> go.Figure:
    """Line chart showing actual AQI over time, optionally with predictions.

    Parameters
    ----------
    df : DataFrame
        Must have a date column and at least one value column.
    """
    fig = go.Figure()

    if actual_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[actual_col],
                mode="lines+markers",
                name="Actual AQI",
                line={"color": "#1f77b4", "width": 3},
                marker={"size": 6},
            )
        )

    if predicted_col and predicted_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[predicted_col],
                mode="lines+markers",
                name="Predicted AQI",
                line={"color": "#ff7f0e", "width": 3, "dash": "dash"},
                marker={"size": 6},
            )
        )

    # Add AQI category bands
    for y_val, color, label in [
        (50, "rgba(0,228,0,0.08)", "Good (50)"),
        (100, "rgba(255,255,0,0.08)", "Moderate (100)"),
        (150, "rgba(255,126,0,0.08)", "USG (150)"),
        (200, "rgba(255,0,0,0.08)", "Unhealthy (200)"),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dot",
            line_color=color,
            opacity=0.5,
            annotation_text=label,
            annotation_position="right",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="AQI",
        height=400,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def pollutant_breakdown(
    pollutants: dict,
    title: str = "Pollutant Breakdown",
) -> go.Figure:
    """Horizontal bar chart of pollutant values.

    Parameters
    ----------
    pollutants : dict
        {pollutant_name: concentration_value, ...}
    """
    names = list(pollutants.keys())
    values = list(pollutants.values())

    # Color bars by severity
    pollutant_limits = {
        "pm25": (0, 25, 50, 75, 100),  # µg/m³
        "pm10": (0, 50, 100, 200, 300),
        "o3": (0, 100, 150, 200, 300),
        "no2": (0, 100, 200, 300, 400),
        "so2": (0, 100, 200, 350, 500),
    }

    colors = []
    for name, val in zip(names, values):
        if val is None or pd.isna(val):
            colors.append("#ccc")
            continue
        limits = pollutant_limits.get(name.lower(), (0, 100, 200, 300, 400))
        if val <= limits[1]:
            colors.append("#00E400")
        elif val <= limits[2]:
            colors.append("#FF7E00")
        elif val <= limits[3]:
            colors.append("#FF0000")
        else:
            colors.append("#7E0023")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}" if v is not None and not pd.isna(v) else "N/A" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Concentration (µg/m³)",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def model_comparison_chart(
    metrics: dict,
    metric_key: str = "avg_rmse",
    title: str = "Model Comparison — RMSE",
) -> go.Figure:
    """Grouped bar chart comparing models on a given metric.

    Parameters
    ----------
    metrics : dict
        {model_name: {metric_key: value, ...}, ...}
    """
    models = list(metrics.keys())
    values = [m.get(metric_key, 0) for m in metrics.values()]

    fig = go.Figure(
        go.Bar(
            x=models,
            y=values,
            marker_color=px.colors.qualitative.Set2[:len(models)],
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric_key.upper(),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_beeswarm(shap_values, feature_names: List[str], title: str = "SHAP Feature Importance") -> go.Figure:
    """Simplified beeswarm-style SHAP summary using a horizontal bar chart.

    For SHAP matrices, we show mean(|SHAP|) per feature.
    """
    import numpy as np
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-15:]  # top 15

    fig = go.Figure(
        go.Bar(
            x=mean_abs[idx],
            y=[feature_names[i] for i in idx],
            orientation="h",
            marker_color=px.colors.sequential.Viridis[:len(idx)],
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP Value|",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
