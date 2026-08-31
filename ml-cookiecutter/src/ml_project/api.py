"""
FastAPI application for the Inference Layer.

Uses the InferencePipeline to load models from MLflow and serve predictions.
"""

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from ml_project import __version__
from ml_project.pipelines import InferencePipeline

# Global pipeline instance
_inference_pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize inference pipeline on startup."""
    global _inference_pipeline
    # In a real FTI system, we'd load a specific version/alias from the Registry
    _inference_pipeline = InferencePipeline()
    yield
    _inference_pipeline = None


app = FastAPI(
    title="ML Project Inference API",
    description="FTI Architecture - Inference Layer",
    version=__version__,
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Request body for predictions."""

    features: list[dict[str, Any]] = Field(min_length=1)  # List of records

    model_config = {
        "json_schema_extra": {
            "example": {"features": [{"feature_1": 1.0, "feature_2": 2.0}]}
        }
    }


class PredictionResponse(BaseModel):
    """Response body for predictions."""

    predictions: list[Any]


@app.get("/health")
async def health():
    return {"status": "healthy", "version": __version__}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Run inference using the production model."""
    if _inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(request.features)
        predictions = _inference_pipeline.predict(df)
        return PredictionResponse(predictions=predictions.tolist())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
