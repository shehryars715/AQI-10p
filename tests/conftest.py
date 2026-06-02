"""Shared pytest fixtures and mocks for the AQI predictor test suite."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Sample data generators ────────────────────────────────────────


@pytest.fixture
def sample_aqicn_response() -> Dict[str, Any]:
    """Return a realistic AQICN API response for Delhi."""
    return {
        "status": "ok",
        "data": {
            "aqi": 155,
            "idx": 7648,
            "city": {"name": "Delhi", "url": "https://aqicn.org/city/delhi/"},
            "iaqi": {
                "pm25": {"v": 68.2},
                "pm10": {"v": 120.5},
                "o3": {"v": 45.3},
                "no2": {"v": 32.1},
                "so2": {"v": 8.7},
                "co": {"v": 12.4},
            },
            "time": {"s": "2026-06-03 10:00:00", "tz": "+05:30"},
        },
    }


@pytest.fixture
def sample_openweather_response() -> Dict[str, Any]:
    """Return a realistic OpenWeather API response for Delhi."""
    return {
        "main": {"temp": 32.5, "humidity": 45, "pressure": 1013},
        "wind": {"speed": 3.6, "deg": 180},
        "weather": [{"main": "Haze", "description": "haze"}],
        "dt": 1749000000,
        "name": "Delhi",
    }


@pytest.fixture
def sample_raw_aqi_df() -> pd.DataFrame:
    """Return a DataFrame shaped like the parsed AQICN response."""
    return pd.DataFrame(
        {
            "city": ["Delhi"],
            "timestamp": [pd.Timestamp("2026-06-03 10:00:00")],
            "aqi": [155],
            "pm25": [68.2],
            "pm10": [120.5],
            "o3": [45.3],
            "no2": [32.1],
            "so2": [8.7],
            "co": [12.4],
        }
    )


@pytest.fixture
def sample_raw_weather_df() -> pd.DataFrame:
    """Return a DataFrame shaped like the parsed OpenWeather response."""
    return pd.DataFrame(
        {
            "city": ["Delhi"],
            "timestamp": [pd.Timestamp("2026-06-03 10:00:00")],
            "temperature": [32.5],
            "humidity": [45],
            "pressure": [1013],
            "wind_speed": [3.6],
            "wind_deg": [180],
            "weather_main": ["Haze"],
            "weather_description": ["haze"],
        }
    )


@pytest.fixture
def sample_engineered_df() -> pd.DataFrame:
    """Return a DataFrame with pre-computed engineered features."""
    return pd.DataFrame(
        {
            "city": ["Delhi"],
            "timestamp": [pd.Timestamp("2026-06-03 10:00:00")],
            "hour": [10],
            "day_of_week": [2],  # Wednesday
            "month": [6],
            "season": ["Summer"],
            "is_weekend": [0],
            "aqi_lag_1h": [148.0],
            "aqi_lag_6h": [142.0],
            "aqi_lag_24h": [135.0],
            "aqi_lag_48h": [130.0],
            "aqi_change_rate_1h": [0.047],
            "aqi_rolling_mean_6h": [145.0],
            "aqi_rolling_mean_24h": [140.0],
            "aqi_category": ["Unhealthy"],
        }
    )


@pytest.fixture
def sample_historical_df() -> pd.DataFrame:
    """Return 72 hours of synthetic AQI data for testing transformers."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-06-01 00:00", periods=72, freq="h")
    base_aqi = 100 + rng.normal(0, 15, size=72).cumsum() * 0.3
    base_aqi = np.clip(base_aqi, 30, 300)

    return pd.DataFrame(
        {
            "city": "Delhi",
            "timestamp": timestamps,
            "aqi": base_aqi,
            "pm25": base_aqi * 0.45,
            "pm10": base_aqi * 0.8,
            "temperature": 28 + rng.normal(0, 2, size=72),
            "humidity": 50 + rng.normal(0, 10, size=72),
        }
    )


# ── Mock helpers ───────────────────────────────────────────────────


@pytest.fixture
def mock_requests_get():
    """Patch requests.get to return canned responses."""
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_hopsworks_project():
    """Return a MagicMock that mimics a Hopsworks project."""
    project = MagicMock()
    fs = MagicMock()
    project.get_feature_store.return_value = fs
    return project, fs
