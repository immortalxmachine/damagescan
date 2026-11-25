"""Shared inference used by both the FastAPI service and the Gradio demo.

Loading the model once and exposing a single ``predict`` function means the two
front ends cannot drift apart: they classify images identically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES
from .data import preprocess_image


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxPredictor:
    """Run an exported ONNX classifier on PIL images."""

    def __init__(self, model_path: str | Path, image_size: int = 224):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.image_size = image_size
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Read the class order written beside the model at export time; fall
        # back to the project default if the sidecar is missing.
        classes_path = self.model_path.with_suffix(".classes.json")
        if classes_path.exists():
            self.class_names = json.loads(classes_path.read_text())
        else:
            self.class_names = list(CLASS_NAMES)

    def predict(self, image: Image.Image) -> dict:
        """Return {label, confidence, probabilities} for one image."""
        x = preprocess_image(image, self.image_size).numpy()
        logits = self.session.run(None, {self.input_name: x})[0]
        probs = _softmax(logits)[0]
        top = int(probs.argmax())
        return {
            "label": self.class_names[top],
            "confidence": round(float(probs[top]), 4),
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(self.class_names, probs, strict=False)
            },
        }
