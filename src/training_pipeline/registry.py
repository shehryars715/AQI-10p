"""Hopsworks Model Registry operations — save, retrieve, and manage models."""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.hopsworks_setup import get_project

logger = logging.getLogger(__name__)

MODEL_NAME = settings.model_registry_name


def _get_or_create_model(project, name: str = MODEL_NAME):
    """Get or create a model entry in the Model Registry."""
    mr = project.get_model_registry()
    try:
        model = mr.get_model(name=name)
        logger.debug("Found existing model: %s", name)
    except Exception:
        model = mr.python.create_model(name=name)
        logger.info("Created new model: %s", name)
    return mr, model


def register_model(
    trained_model: Any,
    model_type: str,
    metrics: Dict[str, float],
    preprocessor: StandardScaler,
    feature_names: list[str],
    hyperparams: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> int:
    """Save a trained model + artifacts to the Hopsworks Model Registry.

    Returns the new model version number.
    """
    project = get_project()
    mr, model = _get_or_create_model(project)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # Save model artifact
        if model_type == "lstm":
            model_path = tmpdir / "model.keras"
            trained_model.save(model_path)
        else:
            model_path = tmpdir / "model.joblib"
            joblib.dump(trained_model, model_path)

        # Save preprocessor
        preprocessor_path = tmpdir / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)

        # Save metadata
        metadata = {
            "model_type": model_type,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "feature_names": feature_names,
            "horizons": settings.prediction_horizons,
            "hyperparameters": hyperparams or {},
            "metrics": metrics,
            "description": description,
        }
        metadata_path = tmpdir / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Register
        model_version = model.save(
            model_path=str(tmpdir),
            await_registration=360,  # seconds
        )
        logger.info(
            "Registered '%s' v%d (%s): avg_rmse=%.2f",
            MODEL_NAME,
            model_version.version,
            model_type,
            metrics.get("avg_rmse", float("nan")),
        )
        return model_version.version


def set_production_tag(version: int) -> None:
    """Tag *version* as 'production' in the Model Registry."""
    project = get_project()
    mr = project.get_model_registry()
    model = mr.get_model(name=MODEL_NAME, version=version)
    model.set_tag("stage", "production")
    logger.info("Set model '%s' v%d → production", MODEL_NAME, version)


def get_production_model() -> Any:
    """Download and load the production model + its preprocessor.

    Returns (model_object, preprocessor, metadata_dict).
    """
    project = get_project()
    mr = project.get_model_registry()
    model = mr.get_model(name=MODEL_NAME, version=None)  # latest

    # Find the version tagged "production"
    versions = mr.get_models(name=MODEL_NAME)
    prod_version = None
    for v in versions:
        tags = v.tags if hasattr(v, "tags") else {}
        if tags.get("stage") == "production":
            prod_version = v
            break

    if prod_version is None:
        # Fall back to latest
        prod_version = versions[-1] if versions else None

    if prod_version is None:
        raise RuntimeError(f"No model registered under '{MODEL_NAME}'")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        prod_version.download(str(tmpdir))

        # Load metadata
        metadata_path = tmpdir / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

        # Load preprocessor
        preprocessor_path = tmpdir / "preprocessor.joblib"
        preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

        # Load model
        model_type = metadata.get("model_type", "sklearn")
        if model_type == "lstm":
            import tensorflow as tf
            model_obj = tf.keras.models.load_model(tmpdir / "model.keras")
        else:
            model_obj = joblib.load(tmpdir / "model.joblib")

        logger.info(
            "Loaded production model: %s v%d (%s)",
            MODEL_NAME,
            prod_version.version,
            model_type,
        )
        return model_obj, preprocessor, metadata
