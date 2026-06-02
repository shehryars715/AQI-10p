"""FastAPI application entry point.

Usage:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import get_model_cache, router
from src.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the production model from Hopsworks on startup (if configured)."""
    cache = get_model_cache()

    if settings.hopsworks_api_key:
        try:
            from src.training_pipeline.registry import get_production_model

            model_obj, preprocessor, metadata = get_production_model()
            cache["model_obj"] = model_obj
            cache["preprocessor"] = preprocessor
            cache["metadata"] = metadata
            logger.info(
                "Loaded production model: %s v%s (%s)",
                metadata.get("model_name", "unknown"),
                metadata.get("version", "?"),
                metadata.get("model_type", "unknown"),
            )
        except Exception:
            logger.warning(
                "Could not load model from Hopsworks — API will use fallback predictions"
            )
    else:
        logger.info(
            "HOPWORKS_API_KEY not set — API will use synthetic fallback predictions"
        )

    yield

    # Shutdown
    cache.clear()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Pearls AQI Predictor API",
    description="Real-time AQI forecasting for Indian cities",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "app": "Pearls AQI Predictor",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
