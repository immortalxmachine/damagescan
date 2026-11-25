"""Tests for preprocessing, transforms and the stratified split."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from damagescan.data import build_transforms, preprocess_image
from damagescan.synthetic import generate_dataset, make_synthetic_image

# Make scripts/ importable so we can test make_splits directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_splits import make_splits  # noqa: E402


def test_eval_transform_output_shape_and_range():
    """Eval transform yields a (3, 224, 224) float tensor."""
    img = make_synthetic_image("intact", seed=0)
    tfm = build_transforms(train=False, image_size=224)
    out = tfm(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_preprocess_image_adds_batch_dim():
    """preprocess_image returns a batched (1, 3, 224, 224) tensor."""
    img = make_synthetic_image("damaged", seed=3)
    batch = preprocess_image(img)
    assert batch.shape == (1, 3, 224, 224)


def test_train_transform_is_random():
    """Augmentation should make two draws of the same image differ."""
    img = make_synthetic_image("damaged", seed=5)
    tfm = build_transforms(train=True)
    a, b = tfm(img), tfm(img)
    assert not torch.allclose(a, b)


def test_stratified_split_preserves_classes_and_ratios(tmp_path):
    """Every class appears in every split with roughly the right proportions."""
    raw = tmp_path / "raw"
    generate_dataset(raw, n_per_class=20)
    dst = tmp_path / "processed"

    counts = make_splits(src=raw, dst=dst, ratios=(0.7, 0.15, 0.15), seed=1)

    for cls in ("damaged", "intact"):
        assert counts[cls]["total"] == 20
        # 70/15/15 of 20 -> 14/3/3 (int() truncates; remainder lands in test).
        assert counts[cls]["train"] == 14
        assert (dst / "train" / cls).exists()
        assert (dst / "val" / cls).exists()
        assert (dst / "test" / cls).exists()
