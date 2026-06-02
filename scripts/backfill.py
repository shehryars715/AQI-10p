"""Historical data backfill — run the feature pipeline for past dates.

Usage:
    python scripts/backfill.py --days 90          # fill last 90 days
    python scripts/backfill.py --start 2026-01-01 # from a specific date

Notes:
    - AQICN free tier *does not* provide historical data; this script populates
      the feature store by fetching current data and creating a training base.
    - For true historical backfill, you would need a paid AQICN plan or an
      alternate historical dataset (e.g., CPCB, OpenAQ).
    - This script primarily sets up the Feature Group schemas and fills with
      whatever data the APIs can provide.
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("backfill")


def main() -> None:
    parser = argparse.ArgumentParser(description="AQI feature backfill")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of past days to run for (default: 7)",
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not settings.hopsworks_api_key:
        logger.error("HOPWORKS_API_KEY is not set — cannot backfill without Hopsworks")
        sys.exit(1)

    # For the free tier, we can only fetch *current* data.
    # The backfill runs once now to populate at least one snapshot.
    logger.info("Backfill: fetching current data for %s", settings.cities)

    aqi_df, weather_df = fetch_all_cities()
    engineered_df = transform(aqi_df, weather_df)

    if aqi_df.empty:
        logger.warning("No data fetched — exiting")
        return

    fs = get_feature_store()
    insert_all(fs, aqi_df, weather_df, engineered_df)

    logger.info(
        "Backfill complete.  To build a training dataset over time, "
        "let the hourly feature pipeline accumulate data.  "
        "Run this script again after more data has been collected."
    )


if __name__ == "__main__":
    main()
