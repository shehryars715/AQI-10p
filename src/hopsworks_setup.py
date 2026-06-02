"""Hopsworks connection helpers — singleton pattern to avoid re-logging in.

Usage:
    from src.hopsworks_setup import get_project, get_feature_store
    fs = get_feature_store()
"""

from __future__ import annotations

import hopsworks
from hsfs.feature_store import FeatureStore

from src.config import settings

_project: hopsworks.project.Project | None = None
_feature_store: FeatureStore | None = None


def get_project() -> hopsworks.project.Project:
    """Return a cached Hopsworks project handle."""
    global _project
    if _project is None:
        _project = hopsworks.login(
            api_key_value=settings.hopsworks_api_key,
            project=settings.hopsworks_project_name,
        )
    return _project


def get_feature_store() -> FeatureStore:
    """Return a cached Hopsworks Feature Store handle."""
    global _feature_store
    if _feature_store is None:
        _feature_store = get_project().get_feature_store()
    return _feature_store


def reset_connection() -> None:
    """Reset cached handles (useful for testing)."""
    global _project, _feature_store
    _project = None
    _feature_store = None
