"""FastAPI prediction service with health and Prometheus endpoints."""

import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from heart_disease.data import ROOT
from heart_disease.pipeline import FEATURES

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "heart_disease_pipeline.joblib"))
PREDICTIONS = Counter("heart_disease_predictions_total", "Predictions served", ["prediction"])
PREDICTION_SECONDS = Histogram("heart_disease_prediction_seconds", "Prediction latency")
LOGGER = logging.getLogger("heart_disease.api")
if not LOGGER.handlers:
    # uvicorn configures only its own loggers, so INFO records would otherwise be dropped.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)


class HeartDiseaseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    disease_probability: float


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    app = FastAPI(title="Heart Disease Classifier", version="0.1.0")
    app.state.model = joblib.load(model_path) if Path(model_path).is_file() else None

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        LOGGER.info(
            "api_request method=%s path=%s status_code=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.get("/health")
    def health() -> dict:
        loaded = app.state.model is not None
        if not loaded:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ok", "model_loaded": True}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: HeartDiseaseFeatures, request: Request) -> PredictionResponse:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        values = payload.model_dump()
        frame = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
        started = time.perf_counter()
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1])
        PREDICTION_SECONDS.observe(time.perf_counter() - started)
        PREDICTIONS.labels(prediction=str(prediction)).inc()
        return PredictionResponse(prediction=prediction, disease_probability=probability)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()