"""Historical data backfill using Open-Meteo (free, no API key).

Open-Meteo provides historical air quality + weather data for any date range.
This script backfills data to aqi_data.csv so you can train a model immediately.

Usage:
    python backfill.py                  # last 90 days
    python backfill.py --days 30         # last 30 days
    python backfill.py --start 2026-01-01 --end 2026-06-01
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# Lahore coordinates
LAT = 31.5497
LON = 74.3436
CITY = "Lahore"

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 (µg/m³) to US EPA AQI using the standard breakpoint formula."""
    if pd.isna(pm25):
        return np.nan
    # EPA breakpoints: (low_conc, high_conc, low_aqi, high_aqi)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, a_low, a_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return ((a_high - a_low) / (c_high - c_low)) * (pm25 - c_low) + a_low
    return 500  # cap at 500


def fetch_historical_aqi(start: str, end: str) -> pd.DataFrame:
    """Fetch hourly air quality data from Open-Meteo."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi",
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(AQ_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "pm25": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "co": hourly["carbon_monoxide"],
        "no2": hourly["nitrogen_dioxide"],
        "so2": hourly["sulphur_dioxide"],
        "o3": hourly["ozone"],
        "aqi": [pm25_to_aqi(v) for v in hourly["pm2_5"]],
    })
    df["city"] = CITY
    return df


def fetch_historical_weather(start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo archive."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(WEATHER_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": np.nan,  # not in free archive
        "wind_speed": hourly["wind_speed_10m"],
        "wind_deg": hourly["wind_direction_10m"],
        "weather": "Unknown",
    })
    return df


def backfill(start: str, end: str) -> pd.DataFrame:
    """Fetch and merge historical AQI + weather data."""
    print(f"Fetching {start} -> {end}...")
    aqi_df = fetch_historical_aqi(start, end)
    weather_df = fetch_historical_weather(start, end)

    # Merge on timestamp
    merged = aqi_df.merge(weather_df, on="timestamp", how="left")
    merged["hour"] = merged["timestamp"].dt.hour
    merged["day_of_week"] = merged["timestamp"].dt.dayofweek
    merged["month"] = merged["timestamp"].dt.month

    print(f"  Got {len(merged)} rows")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Backfill AQI data from Open-Meteo")
    parser.add_argument("--days", type=int, default=90,
                        help="Days to backfill (default: 90)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=args.days + 1)).strftime("%Y-%m-%d")

    df = backfill(start, end)
    df.to_csv("aqi_data.csv", index=False)
    print(f"Saved {len(df)} rows to aqi_data.csv")
    print("Ready to train: python train.py")


if __name__ == "__main__":
    main()
