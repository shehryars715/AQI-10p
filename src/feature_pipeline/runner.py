"""Feature pipeline runner — orchestrates fetch → transform → load.

Intended to be called from:
  - GitHub Actions (hourly cron)
  - CLI: ``python src/feature_pipeline/runner.py``
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from src.config import settings
from src.feature_pipeline.fetcher import fetch_all_cities
from src.feature_pipeline.loader import insert_all
from src.feature_pipeline.transformer import transform
from src.hopsworks_setup import get_feature_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_pipeline")


def run() -> None:
    """Execute the full feature pipeline."""
    logger.info(
        "Starting feature pipeline for cities: %s at %s",
        settings.cities,
        datetime.now(timezone.utc).isoformat(),
    )

    # 1. Fetch
    try:
        aqi_df, weather_df = fetch_all_cities()
    except Exception:
        logger.exception("Fatal: data fetching failed")
        sys.exit(1)

    # 2. Transform
    try:
        engineered_df = transform(aqi_df, weather_df)
    except Exception:
        logger.exception("Fatal: feature engineering failed")
        sys.exit(1)

    # 3. Load (only if Hopsworks credentials are configured)
    if not settings.hopsworks_api_key:
        logger.warning(
            "HOPWORKS_API_KEY not set — skipping Feature Store upload.\n"
            "Fetched %d AQI rows and %d weather rows; would have loaded %d engineered rows.",
            len(aqi_df),
            len(weather_df),
            len(engineered_df),
        )
        return

    try:
        fs = get_feature_store()
        insert_all(fs, aqi_df, weather_df, engineered_df)
    except Exception:
        logger.exception("Fatal: Feature Store upload failed")
        sys.exit(1)

    logger.info("Feature pipeline completed successfully")


if __name__ == "__main__":
    run()
