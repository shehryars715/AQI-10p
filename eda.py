"""Exploratory Data Analysis — identify trends & patterns in AQI data.

Usage: python eda.py

Outputs summary statistics, seasonal patterns, diurnal cycles,
and pollution correlations from aqi_data.csv.
"""

import os
import numpy as np
import pandas as pd

DATA_FILE = "aqi_data.csv"


def load():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run backfill.py first.")
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date
    return df


def summary(df: pd.DataFrame):
    print("=" * 50)
    print("  SUMMARY STATISTICS")
    print("=" * 50)
    stats = df[["aqi", "pm25", "pm10", "temperature", "humidity"]].describe()
    print(stats.round(1))
    print()


def trends(df: pd.DataFrame):
    print("=" * 50)
    print("  MONTHLY AQI TRENDS")
    print("=" * 50)
    monthly = df.groupby("month")["aqi"].agg(["mean", "max", "min", "std"]).round(1)
    print(monthly)
    print()

    print("  WEEKLY PATTERN (day of week)")
    print("  Mon=0 ... Sun=6")
    weekly = df.groupby("day_of_week")["aqi"].agg(["mean", "max", "min"]).round(1)
    print(weekly)
    print()

    print("  DIURNAL CYCLE (hour of day)")
    hourly = df.groupby("hour")["aqi"].agg(["mean", "max", "min"]).round(1)
    print(hourly.to_string())
    print()


def pollution_breakdown(df: pd.DataFrame):
    print("=" * 50)
    print("  POLLUTION CORRELATIONS")
    print("=" * 50)
    cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
            "temperature", "humidity", "wind_speed"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()["aqi"].drop("aqi").sort_values(ascending=False)
    print(corr.round(3))
    print()


def alerts_summary(df: pd.DataFrame):
    print("=" * 50)
    print("  AQI CATEGORY DISTRIBUTION")
    print("=" * 50)

    def cat(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "USG"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        return "Hazardous"

    df["category"] = df["aqi"].apply(cat)
    counts = df["category"].value_counts()
    total = len(df)
    for c in ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]:
        n = counts.get(c, 0)
        pct = n / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {c:<20s}: {n:>5d} ({pct:>5.1f}%) {bar}")
    print()

    # Worst days
    print("  TOP 10 WORST AQI DAYS:")
    top = df.nlargest(10, "aqi")[["timestamp", "aqi", "pm25", "category"]]
    for _, row in top.iterrows():
        print(f"    {row['timestamp']}  AQI={row['aqi']:.0f}  PM2.5={row['pm25']:.0f}  [{row['category']}]")
    print()


def main():
    df = load()
    print(f"\nLoaded {len(df):,} rows across {df['date'].nunique():,} days\n")
    summary(df)
    trends(df)
    pollution_breakdown(df)
    alerts_summary(df)


if __name__ == "__main__":
    main()
