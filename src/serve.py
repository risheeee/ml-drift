import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import joblib
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

import features as feat_module

logger = logging.getLogger(__name__)
MLFLOW_URI = "http://localhost:5000/"


class Transaction(BaseModel):
    Time: float; Amount: float = Field(..., ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence_tier: str
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    batch_size: int
    total_latency_ms: float


class ModelState:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.model_version = "none"

    def load(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        try:
            self.model = mlflow.sklearn.load_model("models:/fraud-detector/Production")
            self.pipeline = feat_module.load_pipeline()
            versions = mlflow.MlflowClient().get_latest_versions("fraud-detector", stages=["Production"])
            self.model_version = versions[0].version if versions else "unknown"
            logger.info(f"Loaded Production model v{self.model_version}")
        except Exception as e:
            logger.error(f"Registry load failed: {e}. Falling back to local joblib.")
            model_files = sorted(Path("models").glob("model_*.joblib"))
            if model_files:
                self.model = joblib.load(model_files[-1])
                self.pipeline = feat_module.load_pipeline()
                self.model_version = model_files[-1].stem
                logger.info(f"Loaded local model: {model_files[-1].name}")
            else:
                raise RuntimeError("No model available. Run training first.")

    def reload(self):
        logger.info("Reloading model...")
        self.load()


_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.load()
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


def _confidence_tier(prob: float) -> str:
    return "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.4 else "LOW"


@app.get("/health")
def health():
    return {
        "status": "ok" if _state.model else "degraded",
        "model_version": _state.model_version,
        "model_loaded": _state.model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if not _state.model:
        raise HTTPException(503, "Model not loaded")
    df = pd.DataFrame([transaction.model_dump()]).assign(Class=0)
    start = time.perf_counter()
    try:
        X, _ = feat_module.transform(df, _state.pipeline)
    except ValueError as e:
        raise HTTPException(422, f"Feature mismatch: {e}")
    prob = float(_state.model.predict_proba(X)[0, 1])
    return PredictionResponse(
        fraud_probability=prob,
        is_fraud=prob >= 0.5,
        confidence_tier=_confidence_tier(prob),
        model_version=_state.model_version,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if not _state.model:
        raise HTTPException(503, "Model not loaded")
    df = pd.DataFrame([t.model_dump() for t in request.transactions]).assign(Class=0)
    start = time.perf_counter()
    try:
        X, _ = feat_module.transform(df, _state.pipeline)
    except ValueError as e:
        raise HTTPException(422, f"Feature mismatch: {e}")
    probs = _state.model.predict_proba(X)[:, 1]
    total_ms = (time.perf_counter() - start) * 1000
    return BatchResponse(
        predictions=[
            PredictionResponse(
                fraud_probability=float(p), is_fraud=float(p) >= 0.5,
                confidence_tier=_confidence_tier(float(p)),
                model_version=_state.model_version,
                latency_ms=round(total_ms / len(probs), 3),
            ) for p in probs
        ],
        batch_size=len(probs),
        total_latency_ms=round(total_ms, 3),
    )


@app.post("/model/reload")
def reload_model():
    try:
        _state.reload()
        return {"status": "reloaded", "model_version": _state.model_version}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)