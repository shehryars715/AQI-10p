"""Run both pipelines locally for testing.

Usage:
    python scripts/local_test_pipeline.py          # feature only
    python scripts/local_test_pipeline.py --train  # feature + training
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("local_test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local pipeline test runner")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also run the training pipeline after the feature pipeline",
    )
    args = parser.parse_args()

    logger.info("=== Running Feature Pipeline ===")
    from src.feature_pipeline.runner import run as run_feature

    try:
        run_feature()
    except SystemExit as e:
        if e.code != 0:
            logger.error("Feature pipeline exited with code %d", e.code)
            if not args.train:
                sys.exit(e.code)

    if args.train:
        logger.info("=== Running Training Pipeline ===")
        from src.training_pipeline.runner import run as run_training

        try:
            run_training()
        except SystemExit as e:
            logger.error("Training pipeline exited with code %d", e.code)
            sys.exit(e.code)

    logger.info("All done!")


if __name__ == "__main__":
    main()
