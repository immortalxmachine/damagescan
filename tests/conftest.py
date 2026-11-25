"""Shared pytest fixtures.

The API tests must not depend on a trained model or a GPU, so we build a tiny
untrained CNN, export it to ONNX, and point the service at it. This mirrors the
real serving path (ONNX + onnxruntime) while staying fast and deterministic.
"""
from __future__ import annotations

import io
import json

import pytest
import torch
import torch.nn as nn

from damagescan.synthetic import make_synthetic_image


class TinyNet(nn.Module):
    """A minimal 2-class CNN standing in for a trained model in tests."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, num_classes)

    def forward(self, x):
        x = torch.relu(self.features(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


@pytest.fixture(scope="session")
def tiny_onnx_model(tmp_path_factory) -> str:
    """Export a tiny model to ONNX (+ class sidecar) and return its path."""
    model_dir = tmp_path_factory.mktemp("models")
    onnx_path = model_dir / "tiny.onnx"
    model = TinyNet().eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    onnx_path.with_suffix(".classes.json").write_text(json.dumps(["damaged", "intact"]))
    return str(onnx_path)


@pytest.fixture
def synthetic_image_bytes() -> bytes:
    """A valid JPEG of a synthetic parcel, as upload bytes."""
    img = make_synthetic_image("damaged", seed=1)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tiny_onnx_model, monkeypatch):
    """A FastAPI TestClient wired to the tiny ONNX model."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_PATH", tiny_onnx_model)
    # Import after setting the env var so defaults resolve correctly.
    from app import api

    # Clear the model cache so each test run loads the fixture model fresh.
    api._load_predictor.cache_clear()
    return TestClient(api.app)
