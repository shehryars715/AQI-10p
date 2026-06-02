"""API route handlers for AQI predictions."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    BatchPredictionResponse,
    ForecastHistoryItem,
    ForecastHistoryResponse,
    HealthResponse,
    LatestFeaturesResponse,
    PredictionItem,
    PredictionResponse,
)
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory cache populated on startup (see main.py lifespan)
_model_cache: dict = {
    "model_obj": None,
    "preprocessor": None,
    "metadata": {},
}


def get_model_cache() -> dict:
    return _model_cache


# ── AQI category helper ─────────────────────────────────────────────

def _aqi_category(aqi: float) -> str:
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


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns model status and registered model info."""
    meta = _model_cache.get("metadata", {})
    return HealthResponse(
        status="ok",
        model_name=meta.get("model_name", settings.model_registry_name),
        model_version=meta.get("version"),
        model_type=meta.get("model_type"),
        last_trained=meta.get("training_date"),
        registered_models=meta.get("all_models", {}),
    )


@router.get("/predict", response_model=PredictionResponse)
async def predict(
    city: str = Query(default="Delhi", description="City name"),
):
    """Return 3-day AQI forecast for *city*."""
    model_obj = _model_cache.get("model_obj")
    preprocessor = _model_cache.get("preprocessor")

    now = datetime.now(timezone.utc)

    if model_obj is None or preprocessor is None:
        # Fallback: return synthetic prediction for demo/development
        logger.warning("No model loaded — returning synthetic predictions")
        base_aqi = 120.0 + hash(city) % 80
        predictions = []
        for i, h in enumerate(settings.prediction_horizons):
            pred_date = (now + timedelta(hours=h)).date()
            aqi_val = base_aqi + i * 10
            predictions.append(
                PredictionItem(
                    date=pred_date,
                    aqi_predicted=round(aqi_val, 1),
                    aqi_category=_aqi_category(aqi_val),
                    confidence_interval=[round(aqi_val * 0.9, 1), round(aqi_val * 1.1, 1)],
                )
            )
        return PredictionResponse(
            city=city,
            generated_at=now,
            model={"type": "synthetic_fallback", "version": 0},
            predictions=predictions,
        )

    # Real inference path
    try:
        # Build feature vector from Hopsworks (simplified — uses latest known)
        feature_names = _model_cache["metadata"].get("feature_names", [])
        X = np.zeros((1, len(feature_names)))  # placeholder

        y_pred = model_obj.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)

        predictions = []
        for i, h in enumerate(settings.prediction_horizons):
            pred_date = (now + timedelta(hours=h)).date()
            aqi_val = float(y_pred[0, i])
            predictions.append(
                PredictionItem(
                    date=pred_date,
                    aqi_predicted=round(aqi_val, 1),
                    aqi_category=_aqi_category(aqi_val),
                    confidence_interval=[
                        round(aqi_val * 0.85, 1),
                        round(aqi_val * 1.15, 1),
                    ],
                )
            )

        return PredictionResponse(
            city=city,
            generated_at=now,
            model={
                "name": settings.model_registry_name,
                "type": _model_cache["metadata"].get("model_type", "unknown"),
                "version": _model_cache["metadata"].get("version", "unknown"),
            },
            predictions=predictions,
        )
    except Exception as exc:
        logger.exception("Prediction failed for city=%s", city)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    cities: str = Query(
        default=",".join(settings.cities),
        description="Comma-separated city names",
    ),
):
    """Return 3-day forecasts for multiple cities at once."""
    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    result: dict = {}
    for city in city_list:
        resp = await predict(city=city)
        result[city] = resp.predictions
    return BatchPredictionResponse(predictions=result)


@router.get("/features/latest", response_model=LatestFeaturesResponse)
async def latest_features(
    city: str = Query(default="Delhi"),
    hours: int = Query(default=48, ge=1, le=168),
):
    """Return the latest known feature values for a city."""
    now = datetime.now(timezone.utc)
    # In production this queries Hopsworks; for now returns a placeholder
    return LatestFeaturesResponse(
        city=city,
        timestamp=now,
        aqi=0.0,
        pm25=None,
        pm10=None,
        temperature=None,
        humidity=None,
        wind_speed=None,
        weather_main=None,
    )


@router.get("/models/info")
async def models_info():
    """Return metadata about all registered model versions."""
    meta = _model_cache.get("metadata", {})
    return {
        "models": meta.get("all_models", {}),
        "production_model": meta.get("model_type", "none"),
    }


@router.get("/forecast/history", response_model=ForecastHistoryResponse)
async def forecast_history(
    city: str = Query(default="Delhi"),
    days: int = Query(default=7, ge=1, le=30),
):
    """Return past predictions vs actuals for accuracy tracking."""
    now = date.today()
    history = []
    for i in range(days):
        d = now - timedelta(days=days - i)
        history.append(
            ForecastHistoryItem(
                date=d,
                predicted=0.0,
                actual=None,
                model_type=None,
            )
        )
    return ForecastHistoryResponse(city=city, history=history)
