"""Fetch AQI + weather from free APIs. Returns clean DataFrames."""

import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_API_TOKEN")
OWM_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = ["Lahore", "Karachi", "Islamabad", "Peshawar", "Faisalabad", "Rawalpindi", "Multan", "Quetta"]


def fetch_aqi(city: str) -> dict:
    """Get current AQI + pollutants for a city from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/"
    resp = requests.get(url, params={"token": AQICN_TOKEN}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"AQICN error for {city}: {data.get('data')}")

    iaqi = data["data"].get("iaqi", {})

    def p(key):
        v = iaqi.get(key, {})
        return v.get("v") if isinstance(v, dict) else None

    return {
        "city": city,
        "timestamp": pd.Timestamp(data["data"]["time"]["s"]),
        "aqi": data["data"]["aqi"],
        "pm25": p("pm25"),
        "pm10": p("pm10"),
        "o3": p("o3"),
        "no2": p("no2"),
        "so2": p("so2"),
        "co": p("co"),
    }


def fetch_weather(city: str) -> dict:
    """Get current weather for a city from OpenWeather."""
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": f"{city},PK", "appid": OWM_KEY, "units": "metric"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    return {
        "city": city,
        "timestamp": pd.Timestamp.utcfromtimestamp(data["dt"]),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather": data["weather"][0]["main"],
    }


def fetch_all() -> pd.DataFrame:
    """Fetch AQI + weather for all cities, return merged DataFrame."""
    rows = []
    for city in CITIES:
        try:
            aqi = fetch_aqi(city)
            w = fetch_weather(city)
            merged = {**aqi, **{k: v for k, v in w.items() if k != "city" and k != "timestamp"}}
            rows.append(merged)
        except Exception as e:
            print(f"[WARN] {city}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

    return df


def save_to_csv(df: pd.DataFrame, path: str = "aqi_data.csv") -> None:
    """Append new rows to CSV (creates if not exists)."""
    import os

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["city", "timestamp"], keep="last")
        combined.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"Saved {len(df)} rows to {path}")
