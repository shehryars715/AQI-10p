"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import _model_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset model cache before each test."""
    _model_cache.clear()
    yield
    _model_cache.clear()


class TestHealth:
    def test_returns_ok(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestPredict:
    def test_returns_prediction_for_default_city(self):
        resp = client.get("/api/v1/predict")
        assert resp.status_code == 200
        data = resp.json()
        assert data["city"] == "Delhi"
        assert len(data["predictions"]) == 3
        for pred in data["predictions"]:
            assert "date" in pred
            assert "aqi_predicted" in pred
            assert "aqi_category" in pred

    def test_returns_prediction_for_specific_city(self):
        resp = client.get("/api/v1/predict?city=Mumbai")
        assert resp.status_code == 200
        data = resp.json()
        assert data["city"] == "Mumbai"

    def test_synthetic_fallback_is_used_without_model(self):
        """Without a loaded model, synthetic predictions are returned."""
        resp = client.get("/api/v1/predict?city=Chennai")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"]["type"] == "synthetic_fallback"


class TestBatchPredict:
    def test_returns_multiple_cities(self):
        resp = client.get("/api/v1/predict/batch?cities=Delhi,Mumbai")
        assert resp.status_code == 200
        data = resp.json()
        assert "Delhi" in data["predictions"]
        assert "Mumbai" in data["predictions"]


class TestFeaturesLatest:
    def test_returns_placeholder(self):
        resp = client.get("/api/v1/features/latest?city=Delhi")
        assert resp.status_code == 200
        data = resp.json()
        assert data["city"] == "Delhi"


class TestModelsInfo:
    def test_returns_dict(self):
        resp = client.get("/api/v1/models/info")
        assert resp.status_code == 200


class TestForecastHistory:
    def test_returns_list(self):
        resp = client.get("/api/v1/forecast/history?city=Delhi&days=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["history"]) == 3


class TestRoot:
    def test_returns_app_info(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app"] == "Pearls AQI Predictor"
