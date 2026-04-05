"""
FastAPI application for the Inference Layer.

Uses the InferencePipeline to load models from MLflow and serve predictions.
"""

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_project import __version__
from ml_project.config import settings
from ml_project.pipelines import InferencePipeline

# Global pipeline instance
_inference_pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize inference pipeline on startup."""
    del app
    global _inference_pipeline
    _inference_pipeline = InferencePipeline(model_uri=settings.inference_model_uri)
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

    features: list[dict[str, Any]]

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [{"feature_1": 1.0, "feature_2": 2.0}],
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response body for predictions."""

    predictions: list[Any]


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness endpoint."""
    return {"status": "healthy", "version": __version__}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Run inference using the configured model."""
    if _inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    try:
        df = pd.DataFrame(request.features)
        predictions = _inference_pipeline.predict(df)
        return PredictionResponse(predictions=predictions.tolist())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
