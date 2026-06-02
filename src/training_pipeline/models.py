"""Model factories — Random Forest, Ridge, and LSTM.

Each builder returns an *unfitted* model (or estimator) ready for .fit().
Multi-output: all three models predict AQI at +24 h, +48 h, +72 h simultaneously.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from src.config import settings

logger = logging.getLogger(__name__)


def build_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = 20,
    random_state: int = settings.random_state,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Build a multi-output Random Forest regressor.

    RandomForestRegressor natively supports multi-output
    (one tree predicts all targets simultaneously).
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_ridge(
    alpha: float = 1.0,
    random_state: int = settings.random_state,
) -> MultiOutputRegressor:
    """Build a multi-output Ridge regressor.

    Ridge does NOT natively support multi-output, so we wrap it.
    """
    return MultiOutputRegressor(
        Ridge(alpha=alpha, random_state=random_state),
        n_jobs=-1,
    )


def build_lstm(
    timesteps: int = settings.lstm_timesteps,
    n_features: int = 1,
    n_outputs: int = 3,
    random_state: int = settings.random_state,
):
    """Build a Keras LSTM model for multi-step AQI prediction.

    Parameters
    ----------
    timesteps : int
        Number of past time steps to look back.
    n_features : int
        Number of input features per time step.
    n_outputs : int
        Number of prediction horizons (default 3: +24h, +48h, +72h).

    Returns
    -------
    tf.keras.Model
        Compiled but unfitted model.
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    except ImportError:
        raise ImportError(
            "TensorFlow is not installed. "
            "Install it with: pip install pearls-aqi-predictor[tensorflow]"
        )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                64,
                activation="tanh",
                return_sequences=False,
                input_shape=(timesteps, n_features),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_outputs, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    logger.info("Built LSTM: input=(%d,%d) → output=%d", timesteps, n_features, n_outputs)
    return model


def reshape_for_lstm(
    X: np.ndarray,
    timesteps: int = settings.lstm_timesteps,
) -> np.ndarray:
    """Reshape (samples, features) → (samples - timesteps + 1, timesteps, features).

    This sliding-window transform is needed because the Feature View delivers
    flat rows, not sequences.  Rows at the beginning that don't have enough
    history are dropped.
    """
    if len(X) <= timesteps:
        raise ValueError(
            f"Need at least {timesteps + 1} samples for LSTM, got {len(X)}"
        )

    X_seq = np.lib.stride_tricks.sliding_window_view(X, timesteps, axis=0)
    # sliding_window_view shape: (samples - timesteps + 1, features, timesteps)
    # We need (samples, timesteps, features)
    X_seq = np.transpose(X_seq, (0, 2, 1))
    return X_seq.copy()
