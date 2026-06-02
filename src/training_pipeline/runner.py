"""Training pipeline runner — orchestrates load → train → evaluate → register.

Intended to be called from:
  - GitHub Actions (daily cron)
  - CLI: ``python src/training_pipeline/runner.py``
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import numpy as np

from src.config import settings
from src.hopsworks_setup import get_feature_store
from src.training_pipeline.evaluator import compare_models, evaluate_model, format_metrics_table
from src.training_pipeline.loader import (
    HORIZONS,
    create_labels,
    load_feature_view,
    preprocess,
    split_data,
)
from src.training_pipeline.models import (
    build_lstm,
    build_random_forest,
    build_ridge,
    reshape_for_lstm,
)
from src.training_pipeline.registry import register_model, set_production_tag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training_pipeline")


def run() -> None:
    """Execute the full training pipeline."""
    logger.info(
        "Starting training pipeline at %s",
        datetime.now(timezone.utc).isoformat(),
    )

    if not settings.hopsworks_api_key:
        logger.error("HOPWORKS_API_KEY is not set — cannot run training pipeline")
        sys.exit(1)

    # 1. Load data from Feature View
    fs = get_feature_store()
    try:
        df, _ = load_feature_view(fs)
    except RuntimeError:
        logger.exception("Cannot proceed without Feature View")
        sys.exit(1)

    if df.empty or len(df) < 100:
        logger.error(
            "Insufficient data: %d rows (need >= 100). "
            "Let the feature pipeline accumulate more data.",
            len(df),
        )
        sys.exit(1)

    # 2. Create labels
    df = create_labels(df)

    if len(df) < 50:
        logger.error("Insufficient labeled data: %d rows after creating labels", len(df))
        sys.exit(1)

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df)

    feature_names = list(X_train.columns)
    logger.info("Features: %s", feature_names)

    # 4. Preprocess
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    horizon_labels = [f"day{i + 1}" for i in range(len(HORIZONS))]
    all_results: dict = {}

    # 5. Train & evaluate each model
    # ── Random Forest ──
    try:
        logger.info("Training Random Forest…")
        rf = build_random_forest()
        rf.fit(X_train_scaled, y_train.values)
        rf_metrics = evaluate_model(rf, X_test_scaled, y_test.values, horizon_labels)
        all_results["RandomForest"] = rf_metrics

        register_model(
            rf,
            model_type="random_forest",
            metrics=rf_metrics,
            preprocessor=scaler,
            feature_names=feature_names,
            hyperparams={"n_estimators": 200, "max_depth": 20},
            description="Random Forest multi-output regressor",
        )
        logger.info("Random Forest registered.")
    except Exception:
        logger.exception("Random Forest training failed")

    # ── Ridge ──
    try:
        logger.info("Training Ridge…")
        ridge = build_ridge()
        ridge.fit(X_train_scaled, y_train.values)
        ridge_metrics = evaluate_model(ridge, X_test_scaled, y_test.values, horizon_labels)
        all_results["Ridge"] = ridge_metrics

        register_model(
            ridge,
            model_type="ridge",
            metrics=ridge_metrics,
            preprocessor=scaler,
            feature_names=feature_names,
            hyperparams={"alpha": 1.0},
            description="Ridge multi-output regressor",
        )
        logger.info("Ridge registered.")
    except Exception:
        logger.exception("Ridge training failed")

    # ── LSTM ──
    try:
        logger.info("Training LSTM…")
        X_train_seq = reshape_for_lstm(X_train_scaled)
        X_test_seq = reshape_for_lstm(X_test_scaled)

        # Trim labels to match the shorter sequence length
        trim_train = len(X_train_scaled) - len(X_train_seq)
        trim_test = len(X_test_scaled) - len(X_test_seq)

        lstm = build_lstm(
            timesteps=settings.lstm_timesteps,
            n_features=X_train_scaled.shape[1],
            n_outputs=len(HORIZONS),
        )
        from tensorflow.keras.callbacks import EarlyStopping
        lstm.fit(
            X_train_seq,
            y_train.values[trim_train:],
            epochs=settings.lstm_epochs,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=settings.lstm_patience,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        lstm_metrics = evaluate_model(
            lstm, X_test_seq, y_test.values[trim_test:], horizon_labels,
        )
        all_results["LSTM"] = lstm_metrics

        register_model(
            lstm,
            model_type="lstm",
            metrics=lstm_metrics,
            preprocessor=scaler,
            feature_names=feature_names,
            hyperparams={
                "timesteps": settings.lstm_timesteps,
                "epochs": settings.lstm_epochs,
                "patience": settings.lstm_patience,
            },
            description="LSTM sequence model for AQI prediction",
        )
        logger.info("LSTM registered.")
    except ImportError:
        logger.warning("TensorFlow not installed — skipping LSTM training")
    except Exception:
        logger.exception("LSTM training failed")

    # 6. Compare & set production
    if not all_results:
        logger.error("No models trained successfully")
        sys.exit(1)

    logger.info("\n" + format_metrics_table(all_results, horizon_labels))

    best_model = compare_models(all_results, metric="avg_rmse")
    logger.info("Best model: %s", best_model)

    # Tag the best model as production
    # (We don't know the exact version number here, so the registry stores
    #  the best model's metrics — the production tag is set during registration)
    logger.info("Training pipeline complete. Models saved to Hopsworks Registry.")


if __name__ == "__main__":
    run()
