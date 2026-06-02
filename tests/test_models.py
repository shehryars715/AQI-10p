"""Unit tests for model factories."""

from __future__ import annotations

import numpy as np
import pytest

from src.training_pipeline.models import (
    build_random_forest,
    build_ridge,
    build_lstm,
    reshape_for_lstm,
)


class TestRandomForest:
    def test_builds(self):
        rf = build_random_forest(n_estimators=10, max_depth=5)
        assert rf is not None
        assert rf.n_estimators == 10

    def test_multi_output_prediction(self):
        rf = build_random_forest(n_estimators=10, max_depth=5)
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)  # 3 outputs
        rf.fit(X, y)
        preds = rf.predict(X[:5])
        assert preds.shape == (5, 3)


class TestRidge:
    def test_builds(self):
        ridge = build_ridge(alpha=0.5)
        assert ridge is not None

    def test_multi_output_prediction(self):
        ridge = build_ridge()
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)
        ridge.fit(X, y)
        preds = ridge.predict(X[:5])
        assert preds.shape == (5, 3)


class TestLSTM:
    def test_builds(self):
        try:
            lstm = build_lstm(timesteps=10, n_features=5, n_outputs=3)
            assert lstm is not None
            assert lstm.output_shape == (None, 3)
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_training_loop(self):
        try:
            lstm = build_lstm(timesteps=5, n_features=2, n_outputs=2)
            X = np.random.randn(200, 5, 2)
            y = np.random.randn(200, 2) * 50 + 100
            history = lstm.fit(X, y, epochs=2, batch_size=16, verbose=0)
            assert history is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")


class TestReshapeForLSTM:
    def test_output_shape(self):
        X = np.arange(100).reshape(50, 2).astype(float)
        X_seq = reshape_for_lstm(X, timesteps=10)
        assert X_seq.shape == (41, 10, 2)  # 50 - 10 + 1 = 41

    def test_insufficient_data(self):
        X = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="Need at least"):
            reshape_for_lstm(X, timesteps=10)
