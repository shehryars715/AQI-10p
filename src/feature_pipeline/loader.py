"""Hopsworks Feature Store writer — upserts data into Feature Groups.

Idempotent: re-running the same hour overwrites rather than duplicating.
"""

from __future__ import annotations

import logging

import pandas as pd
from hsfs.feature_store import FeatureStore

from src.config import settings

logger = logging.getLogger(__name__)

FG_AIR_QUALITY = settings.air_quality_fg_name
FG_WEATHER = settings.weather_fg_name
FG_ENGINEERED = settings.engineered_fg_name
FG_VERSION = settings.feature_group_version

PRIMARY_KEYS = ["city", "timestamp"]


def _ensure_feature_group(
    fs: FeatureStore,
    name: str,
    version: int,
    df: pd.DataFrame,
    description: str,
    event_time: str = "timestamp",
) -> object:
    """Get-or-create a Feature Group matching *df*'s schema."""
    try:
        fg = fs.get_feature_group(name=name, version=version)
        logger.debug("Found existing feature group: %s v%d", name, version)
    except Exception:
        fg = fs.create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=PRIMARY_KEYS,
            event_time=event_time,
            online_enabled=True,
        )
        logger.info("Created feature group: %s v%d", name, version)
    return fg


def insert_air_quality(fs: FeatureStore, df: pd.DataFrame) -> None:
    """Upsert *df* into the ``air_quality`` Feature Group."""
    if df.empty:
        logger.warning("Skipping air_quality insert — DataFrame is empty")
        return

    fg = _ensure_feature_group(
        fs,
        FG_AIR_QUALITY,
        FG_VERSION,
        df,
        "Raw AQI and pollutant readings from AQICN",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    logger.info("Inserted %d rows → %s", len(df), FG_AIR_QUALITY)


def insert_weather(fs: FeatureStore, df: pd.DataFrame) -> None:
    """Upsert *df* into the ``weather`` Feature Group."""
    if df.empty:
        logger.warning("Skipping weather insert — DataFrame is empty")
        return

    fg = _ensure_feature_group(
        fs,
        FG_WEATHER,
        FG_VERSION,
        df,
        "Weather readings from OpenWeather",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    logger.info("Inserted %d rows → %s", len(df), FG_WEATHER)


def insert_engineered(fs: FeatureStore, df: pd.DataFrame) -> None:
    """Upsert *df* into the ``engineered_features`` Feature Group."""
    if df.empty:
        logger.warning("Skipping engineered_features insert — DataFrame is empty")
        return

    fg = _ensure_feature_group(
        fs,
        FG_ENGINEERED,
        FG_VERSION,
        df,
        "Engineered features (lags, rolling stats, time features)",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    logger.info("Inserted %d rows → %s", len(df), FG_ENGINEERED)


def insert_all(
    fs: FeatureStore,
    aqi_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
) -> None:
    """Write all three DataFrames to their respective Feature Groups."""
    insert_air_quality(fs, aqi_df)
    insert_weather(fs, weather_df)
    insert_engineered(fs, engineered_df)
