"""API clients for AQICN and OpenWeather.

Each function returns a clean pandas DataFrame with a consistent schema.
Rate-limited calls are retried with exponential backoff (max 3 attempts).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests

from src.config import settings

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
AQICN_BASE_URL = "https://api.waqi.info/feed/{city}/"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
MAX_RETRIES = 3
BACKOFF_SECONDS = 5


def _retry_get(url: str, params: Dict[str, Any]) -> requests.Response:
    """GET with retry + exponential backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "API call failed (attempt %d/%d), retrying in %ds: %s",
                attempt,
                MAX_RETRIES,
                wait,
                exc,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _validate_api_key(key: Optional[str], name: str) -> str:
    """Ensure an API key is configured, raising a clear error if not."""
    if not key:
        raise ValueError(
            f"{name} is not set.  Add it to your .env file or environment."
        )
    return key


# ── AQICN ───────────────────────────────────────────────────────────


def fetch_aqicn(city: str) -> pd.DataFrame:
    """Fetch current AQI data for *city* from the AQICN API.

    Returns a single-row DataFrame with columns:
        city, timestamp, aqi, pm25, pm10, o3, no2, so2, co
    """
    token = _validate_api_key(settings.aqicn_api_token, "AQICN_API_TOKEN")
    resp = _retry_get(f"{AQICN_BASE_URL}{city}/", {"token": token})
    data: Dict[str, Any] = resp.json()

    if data.get("status") != "ok":
        error_msg = data.get("data", "unknown error")
        logger.error("AQICN API error for %s: %s", city, error_msg)
        raise RuntimeError(f"AQICN API returned error for {city}: {error_msg}")

    iaqi = data["data"].get("iaqi", {})

    def _pollutant(key: str) -> Optional[float]:
        val = iaqi.get(key, {})
        if isinstance(val, dict):
            return val.get("v")
        return None

    return pd.DataFrame(
        {
            "city": [city],
            "timestamp": [pd.Timestamp(data["data"]["time"]["s"])],
            "aqi": [data["data"]["aqi"]],
            "pm25": [_pollutant("pm25")],
            "pm10": [_pollutant("pm10")],
            "o3": [_pollutant("o3")],
            "no2": [_pollutant("no2")],
            "so2": [_pollutant("so2")],
            "co": [_pollutant("co")],
        }
    )


# ── OpenWeather ─────────────────────────────────────────────────────


def fetch_openweather(city: str) -> pd.DataFrame:
    """Fetch current weather for *city* from OpenWeather.

    Returns a single-row DataFrame with columns:
        city, timestamp, temperature, humidity, pressure,
        wind_speed, wind_deg, weather_main, weather_description
    """
    api_key = _validate_api_key(settings.openweather_api_key, "OPENWEATHER_API_KEY")
    resp = _retry_get(
        OPENWEATHER_BASE_URL,
        {"q": city + ",IN", "appid": api_key, "units": "metric"},
    )
    data: Dict[str, Any] = resp.json()

    return pd.DataFrame(
        {
            "city": [city],
            "timestamp": [pd.Timestamp.utcfromtimestamp(data["dt"])],
            "temperature": [data["main"]["temp"]],
            "humidity": [data["main"]["humidity"]],
            "pressure": [data["main"]["pressure"]],
            "wind_speed": [data["wind"]["speed"]],
            "wind_deg": [data["wind"]["deg"]],
            "weather_main": [data["weather"][0]["main"]],
            "weather_description": [data["weather"][0]["description"]],
        }
    )


# ── Combined fetcher ────────────────────────────────────────────────


def fetch_all_cities(
    cities: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch AQI + weather for all configured cities.

    Returns (aqi_df, weather_df) — two DataFrames with one row per city.
    """
    cities = cities or settings.cities
    aqi_frames: list[pd.DataFrame] = []
    weather_frames: list[pd.DataFrame] = []

    for city in cities:
        try:
            aqi_frames.append(fetch_aqicn(city))
        except Exception:
            logger.exception("Failed to fetch AQICN data for %s", city)

        try:
            weather_frames.append(fetch_openweather(city))
        except Exception:
            logger.exception("Failed to fetch OpenWeather data for %s", city)

    aqi_df = (
        pd.concat(aqi_frames, ignore_index=True) if aqi_frames else pd.DataFrame()
    )
    weather_df = (
        pd.concat(weather_frames, ignore_index=True)
        if weather_frames
        else pd.DataFrame()
    )

    logger.info(
        "Fetched %d AQI rows and %d weather rows across %d cities",
        len(aqi_df),
        len(weather_df),
        len(cities),
    )
    return aqi_df, weather_df
