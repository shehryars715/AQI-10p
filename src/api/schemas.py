"""Pydantic request/response schemas for the AQI Predictor API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Health ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    model_name: Optional[str] = None
    model_version: Optional[int] = None
    model_type: Optional[str] = None
    last_trained: Optional[datetime] = None
    registered_models: Dict[str, dict] = Field(default_factory=dict)


# ── Prediction ──────────────────────────────────────────────────────


class PredictionItem(BaseModel):
    date: date
    aqi_predicted: float
    aqi_category: str
    confidence_interval: list[float] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    city: str
    generated_at: datetime
    model: dict = Field(default_factory=dict)
    predictions: list[PredictionItem]


class BatchPredictionResponse(BaseModel):
    predictions: Dict[str, list[PredictionItem]]


# ── Features ────────────────────────────────────────────────────────


class LatestFeaturesResponse(BaseModel):
    city: str
    timestamp: datetime
    aqi: float
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    weather_main: Optional[str] = None


# ── Models Info ─────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    name: str
    version: int
    model_type: str
    trained_at: datetime
    metrics: dict
    tags: dict = Field(default_factory=dict)


class ModelsInfoResponse(BaseModel):
    models: list[ModelInfo]


# ── Forecast History ────────────────────────────────────────────────


class ForecastHistoryItem(BaseModel):
    date: date
    predicted: float
    actual: Optional[float] = None
    model_type: Optional[str] = None


class ForecastHistoryResponse(BaseModel):
    city: str
    history: list[ForecastHistoryItem]
