"""Alert components for the dashboard."""

from __future__ import annotations

from typing import List, Optional

import streamlit as st


def hazardous_alert_card(
    day: str,
    aqi: float,
    city: str = "Delhi",
    category: str = "Hazardous",
) -> None:
    """Render a single alert card for a hazardous AQI day.

    Parameters
    ----------
    day : str
        Date label (e.g., "Jun 4, 2026").
    aqi : float
        Predicted AQI.
    city : str
        City name.
    category : str
        EPA category label.
    """
    emoji, border = _alert_style(category)

    with st.container(border=True):
        st.markdown(f"### {emoji} {category.upper()}")
        st.metric(label=f"{city} — {day}", value=f"AQI {aqi:.0f}")
        if aqi > 200:
            st.caption(
                "⚠️ **Health warning:** Everyone may experience serious health effects. "
                "Avoid outdoor activities."
            )
        elif aqi > 150:
            st.caption(
                "⚠️ **Health warning:** Everyone may begin to experience health effects. "
                "Members of sensitive groups may experience more serious effects."
            )
        elif aqi > 100:
            st.caption(
                "⚠️ **Sensitive groups:** People with respiratory or heart conditions, "
                "children, and older adults should limit prolonged outdoor exertion."
            )


def _alert_style(category: str) -> tuple[str, str]:
    """Return (emoji, border_color) for an AQI category."""
    styles = {
        "Hazardous": ("🟤", "#7E0023"),
        "Very Unhealthy": ("🟣", "#8F3F97"),
        "Unhealthy": ("🔴", "#FF0000"),
        "Unhealthy for Sensitive Groups": ("🟠", "#FF7E00"),
        "Moderate": ("🟡", "#FFFF00"),
        "Good": ("🟢", "#00E400"),
    }
    return styles.get(category, ("⚪", "#ccc"))


def city_comparison_table(predictions: dict) -> None:
    """Render a comparison table across multiple cities.

    Parameters
    ----------
    predictions : dict
        {city: [PredictionItem, ...], ...}
    """
    if not predictions:
        st.info("No predictions available yet.")
        return

    # Build table data
    rows = []
    for city, preds in predictions.items():
        row = {"City": city}
        for i, p in enumerate(preds):
            row[f"Day {i + 1}"] = f"{p['aqi_predicted']:.0f} ({p['aqi_category'][:4]})"
        row["Trend"] = _trend_arrow(preds)
        rows.append(row)

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _trend_arrow(preds: list) -> str:
    """Return an arrow indicating the trend over 3 days."""
    if len(preds) < 2:
        return "➡️"
    first = preds[0]["aqi_predicted"]
    last = preds[-1]["aqi_predicted"]
    diff = last - first
    if diff > 20:
        return "↗️ Worsening"
    elif diff < -20:
        return "↘️ Improving"
    else:
        return "➡️ Stable"
