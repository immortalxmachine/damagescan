"""Gradio demo: drag-and-drop a parcel photo and see the prediction.

Run with::

    python -m app.gradio_app

It reuses the same OnnxPredictor as the FastAPI service, so the demo and the
API always agree.
"""
from __future__ import annotations

import os

import gradio as gr
from PIL import Image

from damagescan.config import MODELS_DIR
from damagescan.inference import OnnxPredictor

MODEL_PATH = os.environ.get("MODEL_PATH", str(MODELS_DIR / "resnet18.onnx"))


def _load_predictor() -> OnnxPredictor:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train and export a model first "
            "(see the README), or set MODEL_PATH."
        )
    return OnnxPredictor(MODEL_PATH)


def classify(image: Image.Image) -> dict:
    """Gradio callback: image -> {label: confidence} for the Label widget."""
    predictor = _load_predictor()
    result = predictor.predict(image)
    # gr.Label renders a dict of {class: probability} as a ranked bar chart.
    return result["probabilities"]


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Parcel photo"),
        outputs=gr.Label(num_top_classes=2, label="Prediction"),
        title="DamageScan — parcel damage detection",
        description="Upload a parcel photo; the model predicts intact vs damaged.",
    )


if __name__ == "__main__":
    build_demo().launch()
