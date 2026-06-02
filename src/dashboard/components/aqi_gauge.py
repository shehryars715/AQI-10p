"""Color-coded AQI gauge chart using Plotly."""

from __future__ import annotations

import plotly.graph_objects as go

# EPA AQI color scale
AQI_COLORS = {
    "Good": "#00E400",                           # Green
    "Moderate": "#FFFF00",                        # Yellow
    "Unhealthy for Sensitive Groups": "#FF7E00",  # Orange
    "Unhealthy": "#FF0000",                       # Red
    "Very Unhealthy": "#8F3F97",                  # Purple
    "Hazardous": "#7E0023",                       # Maroon
}

# Thresholds for the gauge segments
THRESHOLDS = [
    (0, 50, "Good", "#00E400"),
    (51, 100, "Moderate", "#FFFF00"),
    (101, 150, "USG", "#FF7E00"),
    (151, 200, "Unhealthy", "#FF0000"),
    (201, 300, "Very Unhealthy", "#8F3F97"),
    (301, 500, "Hazardous", "#7E0023"),
]


def aqi_color(aqi: float) -> str:
    """Return the hex color for an AQI value."""
    for low, high, _, color in THRESHOLDS:
        if low <= aqi <= high:
            return color
    return "#7E0023"  # default: Hazardous


def aqi_gauge(aqi: float, title: str = "Current AQI") -> go.Figure:
    """Build a Plotly indicator/gauge for AQI.

    Parameters
    ----------
    aqi : float
        The AQI value to display.
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=aqi,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 24}},
            delta={"reference": 100, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "darkgray"},
                "bar": {"color": aqi_color(aqi)},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 50], "color": "#00E400"},
                    {"range": [51, 100], "color": "#FFFF00"},
                    {"range": [101, 150], "color": "#FF7E00"},
                    {"range": [151, 200], "color": "#FF0000"},
                    {"range": [201, 300], "color": "#8F3F97"},
                    {"range": [301, 500], "color": "#7E0023"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": aqi,
                },
            },
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "darkgray", "family": "Arial"},
    )

    return fig


def aqi_category_label(aqi: float) -> str:
    """Return the EPA category label for a given AQI."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"
