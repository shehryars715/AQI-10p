"""Training data loader — reads from Hopsworks Feature View, creates labels.

The Feature View joins air_quality + weather + engineered_features.
Labels are created by shifting AQI forward by the configured horizons.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)

FV_NAME = settings.feature_view_name
FV_VERSION = settings.feature_view_version
HORIZONS = settings.prediction_horizons


def load_feature_view(
    fs: FeatureStore,
    name: str = FV_NAME,
    version: int = FV_VERSION,
) -> Tuple[pd.DataFrame, Optional[list[str]]]:
    """Load the joined Feature View as a pandas DataFrame.

    Returns (df, labels) where *labels* is None if not yet defined.

    Raises RuntimeError if the Feature View does not exist.
    """
    try:
        fv = fs.get_feature_view(name=name, version=version)
        query = fv.select_all()
        df = query.read()
        logger.info("Loaded Feature View '%s' v%d: %d rows, %d cols",
                     name, version, len(df), len(df.columns))
        labels = fv.labels if hasattr(fv, 'labels') else None
        return df, labels
    except Exception as exc:
        logger.error("Failed to load Feature View '%s' v%d: %s", name, version, exc)
        raise RuntimeError(
            f"Feature View '{name}' v{version} not found. "
            "Run the feature pipeline first to create it."
        ) from exc


def create_labels(
    df: pd.DataFrame,
    horizons: Optional[list[int]] = None,
    aqi_col: str = "aqi",
    group_col: str = "city",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Create target columns: ``aqi_t+24h``, ``aqi_t+48h``, ``aqi_t+72h``.

    Rows without a valid label for *any* horizon are dropped.
    """
    horizons = horizons or HORIZONS
    df = df.copy().sort_values([group_col, time_col])

    for h in horizons:
        label_col = f"aqi_t+{h}h"
        df[label_col] = df.groupby(group_col)[aqi_col].shift(-h)

    before = len(df)
    label_cols = [f"aqi_t+{h}h" for h in horizons]
    df = df.dropna(subset=label_cols)
    after = len(df)

    logger.info(
        "Created labels for horizons %s: %d → %d rows (dropped %d)",
        horizons, before, after, before - after,
    )
    return df


def split_data(
    df: pd.DataFrame,
    label_cols: Optional[list[str]] = None,
    test_size: float = settings.test_size,
    random_state: int = settings.random_state,
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological train/test split (NO shuffle — prevents time-series leakage).

    Returns (X_train, X_test, y_train, y_test).
    """
    horizons = settings.prediction_horizons
    label_cols = label_cols or [f"aqi_t+{h}h" for h in horizons]

    # Identify feature columns (exclude keys, labels, and non-numeric)
    feature_cols = [
        c for c in df.columns
        if c not in ["city", time_col] + label_cols
        and np.issubdtype(df[c].dtype, np.number)
    ]

    X = df[feature_cols].copy()
    y = df[label_cols].copy()

    # Chronological split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "Split: train=%d rows, test=%d rows, %d features, %d labels",
        len(X_train), len(X_test), len(feature_cols), len(label_cols),
    )
    return X_train, X_test, y_train, y_test


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale numeric features and return numpy arrays + fitted scaler.

    NaN values are filled with column median from training data.
    """
    # Handle NaN
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Preprocessed: scaled %d features", X_train.shape[1])
    return X_train_scaled, X_test_scaled, scaler
