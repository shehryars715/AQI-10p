"""Unit tests for model evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.training_pipeline.evaluator import (
    compare_models,
    evaluate_model,
    format_metrics_table,
)


class TestEvaluateModel:
    def test_perfect_prediction(self):
        """A model that predicts perfectly should have RMSE=0, MAE=0, R²=1."""
        y = np.array([[100, 150, 200], [110, 140, 190], [105, 145, 195]])
        X = np.arange(9).reshape(3, 3)

        class PerfectModel:
            def predict(self, X):
                return y

        metrics = evaluate_model(
            PerfectModel(),
            X,
            y,
            horizon_labels=["day1", "day2", "day3"],
        )
        assert metrics["rmse_day1"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mae_day1"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["avg_rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_returns_averaged_metrics(self):
        model = DummyRegressor(strategy="mean")
        y = np.random.default_rng(42).normal(100, 20, (50, 3))
        X = np.random.default_rng(42).normal(0, 1, (50, 5))
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)
        assert "avg_rmse" in metrics
        assert "avg_mae" in metrics
        assert "avg_r2" in metrics
        assert metrics["avg_rmse"] >= 0


class TestCompareModels:
    def test_picks_best_by_rmse(self):
        results = {
            "ModelA": {"avg_rmse": 18.0, "avg_mae": 12.0},
            "ModelB": {"avg_rmse": 15.0, "avg_mae": 10.0},
            "ModelC": {"avg_rmse": 22.0, "avg_mae": 16.0},
        }
        best = compare_models(results, metric="avg_rmse")
        assert best == "ModelB"

    def test_picks_best_by_r2(self):
        results = {
            "A": {"avg_r2": 0.85},
            "B": {"avg_r2": 0.93},
        }
        best = compare_models(results, metric="avg_r2", lower_is_better=False)
        assert best == "B"

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            compare_models({})

    def test_raises_on_missing_metric(self):
        with pytest.raises(ValueError):
            compare_models({"M": {"mae": 5.0}}, metric="avg_rmse")


class TestFormatMetricsTable:
    def test_returns_string(self):
        results = {
            "RF": {"avg_rmse": 18.2, "avg_mae": 12.1, "avg_r2": 0.91,
                   "rmse_day1": 15.0, "rmse_day2": 18.5, "rmse_day3": 21.0},
        }
        table = format_metrics_table(results, horizon_labels=["day1", "day2", "day3"])
        assert "RF" in table
        assert "18.2" in table
        assert "0.91" in table
