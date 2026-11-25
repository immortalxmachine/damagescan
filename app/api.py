"""FastAPI serving layer for DamageScan.

Endpoints
---------
GET  /health   -> liveness check, reports whether a model is loaded
POST /predict  -> multipart image upload -> {label, confidence, probabilities}

The model path comes from the MODEL_PATH environment variable (set it in .env),
defaulting to the exported ResNet-18. The predictor is resolved through a
FastAPI dependency so tests can inject a tiny dummy model.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from damagescan.config import MODELS_DIR
from damagescan.inference import OnnxPredictor

DEFAULT_MODEL_PATH = str(MODELS_DIR / "resnet18.onnx")

app = FastAPI(
    title="DamageScan API",
    description="Classify a parcel photo as intact or damaged.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _load_predictor(model_path: str) -> OnnxPredictor:
    """Load and cache the ONNX predictor for a given path."""
    return OnnxPredictor(model_path)


def get_predictor() -> OnnxPredictor:
    """FastAPI dependency: resolve the predictor or raise a clean 503.

    Kept separate from the cache so that a missing model yields a 503 Service
    Unavailable (not a 500), and so tests can override this dependency.
    """
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not Path(model_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {model_path}. Train and export a model, or set MODEL_PATH.",
        )
    return _load_predictor(model_path)


@app.get("/health")
def health() -> dict:
    """Report service health and whether a model file is present."""
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    return {"status": "ok", "model_loaded": Path(model_path).exists(), "model_path": model_path}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...), predictor: OnnxPredictor = Depends(get_predictor)
) -> dict:
    """Classify an uploaded image."""
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    # ONNX inference is CPU-bound; run it off the event loop so the server
    # stays responsive under concurrent requests.
    result = await run_in_threadpool(predictor.predict, image)
    result["filename"] = file.filename
    return result
