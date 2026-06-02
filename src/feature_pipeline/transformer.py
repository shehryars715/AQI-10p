"""Feature engineering — pure pandas transforms (no side effects).

Every function takes a DataFrame and returns a DataFrame with new columns.
Designed to be easy to unit test with synthetic data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# US EPA breakpoints for AQI categories
AQI_CATEGORIES: list[tuple[int, int, str]] = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 9999, "Hazardous"),
]


def aqi_category(aqi: float) -> str:
    """Map an AQI value to its EPA category label."""
    for low, high, label in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Hazardous"


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features derived from the timestamp column.

    Expects: ``timestamp`` (datetime).
    Adds: ``hour``, ``day_of_week``, ``month``, ``season``, ``is_weekend``.
    """
    df = df.copy()
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour.astype(int)
    df["day_of_week"] = ts.dt.dayofweek.astype(int)  # Mon=0 … Sun=6
    df["month"] = ts.dt.month.astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Season for Indian subcontinent
    month = df["month"]
    conditions = [
        month.isin([12, 1, 2]),
        month.isin([3, 4, 5]),
        month.isin([6, 7, 8, 9]),
        month.isin([10, 11]),
    ]
    choices = ["Winter", "Summer", "Monsoon", "Autumn"]
    df["season"] = np.select(conditions, choices, default="Unknown")

    return df


def compute_lag_features(
    df: pd.DataFrame,
    value_col: str = "aqi",
    lag_hours: tuple[int, ...] = (1, 6, 24, 48),
    group_col: str = "city",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Add lagged AQI columns: ``aqi_lag_1h``, ``aqi_lag_6h``, …"""
    df = df.copy().sort_values([group_col, time_col])
    for lag in lag_hours:
        col_name = f"{value_col}_lag_{lag}h"
        df[col_name] = df.groupby(group_col)[value_col].shift(lag)
    return df


def compute_rolling_features(
    df: pd.DataFrame,
    value_col: str = "aqi",
    windows: tuple[int, ...] = (6, 24),
    group_col: str = "city",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Add rolling mean AQI columns: ``aqi_rolling_mean_6h``, …"""
    df = df.copy().sort_values([group_col, time_col])
    for w in windows:
        col_name = f"{value_col}_rolling_mean_{w}h"
        df[col_name] = (
            df.groupby(group_col)[value_col]
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def compute_change_rate(
    df: pd.DataFrame,
    value_col: str = "aqi",
    lag_col: str = "aqi_lag_1h",
) -> pd.DataFrame:
    """Add ``aqi_change_rate_1h`` = (AQI - lag_1h) / lag_1h."""
    df = df.copy()
    df["aqi_change_rate_1h"] = np.where(
        df[lag_col].notna() & (df[lag_col] != 0),
        (df[value_col] - df[lag_col]) / df[lag_col],
        np.nan,
    )
    return df


def compute_aqi_category_col(df: pd.DataFrame, aqi_col: str = "aqi") -> pd.DataFrame:
    """Add ``aqi_category`` string column."""
    df = df.copy()
    df["aqi_category"] = df[aqi_col].apply(aqi_category)
    return df


def transform(
    aqi_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    hist_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Parameters
    ----------
    aqi_df : DataFrame
        New AQI readings (1+ rows, must have ``city``, ``timestamp``, ``aqi``).
    weather_df : DataFrame
        New weather readings.
    hist_df : DataFrame, optional
        Historical AQI data for computing accurate lags/rolling stats.
        If not provided, lags are computed on *aqi_df* alone (limited accuracy).

    Returns
    -------
    DataFrame
        Engineered features with the same primary keys (city, timestamp).
    """
    if aqi_df.empty:
        logger.warning("Empty AQI DataFrame passed to transformer — returning empty")
        return pd.DataFrame()

    # Combine new data with historical for accurate lag computation
    if hist_df is not None and not hist_df.empty:
        combined = pd.concat([hist_df[["city", "timestamp", "aqi"]], aqi_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["city", "timestamp"], keep="last")
    else:
        combined = aqi_df[["city", "timestamp", "aqi"]].copy()

    # Time features (on the new data only)
    result = compute_time_features(aqi_df[["city", "timestamp", "aqi"]].copy())

    # Lag + rolling features (computed on combined, then filtered back to new rows)
    combined = compute_lag_features(combined)
    combined = compute_rolling_features(combined)
    combined = compute_change_rate(combined)

    # Merge lag/rolling back onto result
    lag_cols = (
        ["aqi_lag_1h", "aqi_lag_6h", "aqi_lag_24h", "aqi_lag_48h"]
        + ["aqi_rolling_mean_6h", "aqi_rolling_mean_24h"]
        + ["aqi_change_rate_1h"]
    )
    for col in lag_cols:
        if col in combined.columns:
            result = result.merge(
                combined[["city", "timestamp", col]],
                on=["city", "timestamp"],
                how="left",
            )

    result = compute_aqi_category_col(result)

    logger.info("Engineered %d feature rows", len(result))
    return result
