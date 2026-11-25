"""Tests for the FastAPI service, using the tiny ONNX model fixture."""
from __future__ import annotations


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_schema(client, synthetic_image_bytes):
    resp = client.post(
        "/predict",
        files={"file": ("parcel.jpg", synthetic_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("damaged", "intact")
    assert 0.0 <= body["confidence"] <= 1.0
    # Probabilities cover both classes and sum to ~1.
    probs = body["probabilities"]
    assert set(probs) == {"damaged", "intact"}
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_predict_rejects_non_image(client):
    resp = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert resp.status_code == 400
