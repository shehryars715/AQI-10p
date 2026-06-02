"""Train & compare multiple models for AQI prediction at +24h, +48h, +72h.

Models: Random Forest, Ridge Regression, Neural Network (MLP), optional LSTM
Metrics: RMSE, MAE, R² — per horizon and averaged

Usage: python train.py
Outputs: model.joblib (best model), evaluation.csv (metrics for dashboard)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_FILE = "aqi_data.csv"
MODEL_FILE = "model.joblib"
EVAL_FILE = "evaluation.csv"
MIN_ROWS = 72


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run backfill.py first.")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["city", "timestamp"])


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    for lag in [1, 6, 24]:
        df[f"aqi_lag_{lag}h"] = df.groupby("city")["aqi"].shift(lag)
    for window in [6, 24]:
        df[f"aqi_roll_{window}h"] = (
            df.groupby("city")["aqi"]
            .rolling(window, min_periods=1).mean()
            .reset_index(0, drop=True)
        )
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    for horizon in [24, 48, 72]:
        df[f"aqi_t+{horizon}h"] = df.groupby("city")["aqi"].shift(-horizon)
    return df


def evaluate(y_true, y_pred, label: str) -> dict:
    """Compute RMSE, MAE, R² for one prediction horizon."""
    return {
        "horizon": label,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train():
    print("=" * 60)
    print("  AQI Model Training Pipeline")
    print("=" * 60)

    # ── Load & prepare data ──
    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"       {len(df)} rows, {df['city'].nunique()} cities")

    df = create_features(df)
    df = create_labels(df)

    label_cols = ["aqi_t+24h", "aqi_t+48h", "aqi_t+72h"]
    df = df.dropna(subset=label_cols)

    if len(df) < MIN_ROWS:
        print(f"       Need {MIN_ROWS}+ labeled rows, got {len(df)}. Run backfill first.")
        return

    exclude = ["city", "timestamp"] + label_cols + ["aqi"]
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols]
    y = df[label_cols]

    # Fill NaN with column median, then 0 for all-NaN columns (e.g. pressure)
    X = X.fillna(X.median()).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"       Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"       Features: {len(feature_cols)}")

    # ── Train models ──
    print("\n[2/4] Training models...")
    models = {}

    # Random Forest
    print("       Random Forest...", end=" ")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train.values)
    models["RandomForest"] = rf
    print("OK")

    # Ridge Regression
    print("       Ridge Regression...", end=" ")
    ridge = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
    ridge.fit(X_train_s, y_train.values)
    models["Ridge"] = ridge
    print("OK")

    # Neural Network (MLP)
    print("       Neural Network...", end=" ")
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu",
        max_iter=500, early_stopping=True, random_state=42,
    )
    mlp.fit(X_train_s, y_train.values)
    models["NeuralNet"] = mlp
    print("OK")

    # ── Evaluate ──
    print("\n[3/4] Evaluating...")
    all_rows = []
    horizon_labels = ["Day 1 (+24h)", "Day 2 (+48h)", "Day 3 (+72h)"]

    for name, model in models.items():
        y_pred = model.predict(X_test_s)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        for i, label in enumerate(horizon_labels):
            row = evaluate(y_test.values[:, i], y_pred[:, i], label)
            row["model"] = name
            all_rows.append(row)

    eval_df = pd.DataFrame(all_rows)

    # Print comparison table
    print()
    print(f"  {'Model':<15} {'Horizon':<16} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"  {'-'*15} {'-'*16} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in eval_df.iterrows():
        print(f"  {row['model']:<15} {row['horizon']:<16} {row['rmse']:>8.1f} {row['mae']:>8.1f} {row['r2']:>8.3f}")

    # Average across horizons
    print(f"  {'-'*55}")
    for name in models:
        avg = eval_df[eval_df["model"] == name]
        rmse_avg = avg["rmse"].mean()
        mae_avg = avg["mae"].mean()
        r2_avg = avg["r2"].mean()
        print(f"  {name:<15} {'AVERAGE':<16} {rmse_avg:>8.1f} {mae_avg:>8.1f} {r2_avg:>8.3f}")

    # Pick best model (lowest avg RMSE)
    best_name = min(models, key=lambda n: eval_df[eval_df["model"] == n]["rmse"].mean())
    best_rmse = eval_df[eval_df["model"] == best_name]["rmse"].mean()
    print(f"\n  >> Best model: {best_name} (avg RMSE = {best_rmse:.1f})")

    # ── Save ──
    print(f"\n[4/4] Saving...")

    # Feature importance (from Random Forest — always trained)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": models["RandomForest"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv("feature_importance.csv", index=False)

    bundle = {
        "model": models[best_name],
        "scaler": scaler,
        "features": feature_cols,
        "model_name": best_name,
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"       {MODEL_FILE} <- {best_name}")

    eval_df.to_csv(EVAL_FILE, index=False)
    print(f"       {EVAL_FILE} <- metrics for dashboard")
    print(f"       feature_importance.csv <- top features")

    print(f"\n{'='*60}")
    print(f"  Done. Dashboard will now use: {best_name}")
    print(f"{'='*60}")
    return bundle


if __name__ == "__main__":
    train()
