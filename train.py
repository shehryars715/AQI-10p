"""Train a Random Forest to predict AQI at +24h, +48h, +72h.

Usage: python train.py

Reads aqi_data.csv (created by fetcher), trains the model, saves model.joblib.
If you don't have enough data yet, the dashboard falls back to a heuristic.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_FILE = "aqi_data.csv"
MODEL_FILE = "model.joblib"
MIN_ROWS = 72  # need at least 3 days of hourly data


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the fetcher first to collect data.")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["city", "timestamp"])
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features and lag features."""
    df = df.copy()

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Lag features (AQI at t-1h, t-6h, t-24h)
    for lag in [1, 6, 24]:
        df[f"aqi_lag_{lag}h"] = df.groupby("city")["aqi"].shift(lag)

    # Rolling means
    for window in [6, 24]:
        df[f"aqi_roll_{window}h"] = (
            df.groupby("city")["aqi"].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        )

    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add target columns: AQI at +24h, +48h, +72h."""
    for horizon in [24, 48, 72]:
        df[f"aqi_t+{horizon}h"] = df.groupby("city")["aqi"].shift(-horizon)
    return df


def train(path: str = DATA_FILE):
    print("Loading data...")
    df = load_data(path)
    print(f"  {len(df)} rows, {df['city'].nunique()} cities")

    df = create_features(df)
    df = create_labels(df)
    df = df.dropna()

    if len(df) < MIN_ROWS:
        print(f"  Need at least {MIN_ROWS} labeled rows, got {len(df)}. Collect more data first.")
        return None

    # Features (numeric only, no IDs)
    exclude = ["city", "timestamp", "aqi_t+24h", "aqi_t+48h", "aqi_t+72h", "aqi"]
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[["aqi_t+24h", "aqi_t+48h", "aqi_t+72h"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"  Training on {len(X_train)} rows, {len(feature_cols)} features...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train.values)

    # Evaluate
    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    print(f"  Test RMSE: {rmse:.1f} AQI points")

    # Save model + scaler + feature names
    bundle = {"model": model, "scaler": scaler, "features": feature_cols}
    joblib.dump(bundle, MODEL_FILE)
    print(f"  Saved to {MODEL_FILE}")
    return bundle


if __name__ == "__main__":
    train()
