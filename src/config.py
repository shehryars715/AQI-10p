"""Central configuration loaded from environment variables.

All modules import the singleton `settings` instance.  Sensitive values
(AQICN token, OpenWeather key, Hopsworks API key) are read from the
environment / .env file and never hard-coded.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings backed by environment variables.

    Sensitive fields (API keys) are Optional so the config imports even
    when they are not set.  Each module validates the keys it requires.
    """

    # ── Hopsworks ────────────────────────────────────────────────────
    hopsworks_api_key: Optional[str] = None
    hopsworks_project_name: str = "Pearls_AQI"

    # ── External APIs ────────────────────────────────────────────────
    aqicn_api_token: Optional[str] = None
    openweather_api_key: Optional[str] = None

    # ── Cities ───────────────────────────────────────────────────────
    cities: List[str] = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]

    # ── Feature Store ────────────────────────────────────────────────
    feature_group_version: int = 1
    feature_view_name: str = "aqi_prediction_view"
    feature_view_version: int = 1

    # ── Feature Group names ──────────────────────────────────────────
    air_quality_fg_name: str = "air_quality"
    weather_fg_name: str = "weather"
    engineered_fg_name: str = "engineered_features"

    # ── Model Registry ───────────────────────────────────────────────
    model_registry_name: str = "aqi_predictor"

    # ── Training ─────────────────────────────────────────────────────
    prediction_horizons: List[int] = [24, 48, 72]  # +24 h, +48 h, +72 h
    test_size: float = 0.2
    random_state: int = 42
    lstm_timesteps: int = 48  # 2 full diurnal cycles
    lstm_epochs: int = 100
    lstm_patience: int = 10  # early stopping

    # ── Serving ──────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_refresh_seconds: int = 300

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton — import this everywhere
settings = Settings()
