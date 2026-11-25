"""Central configuration: paths, class names and default hyperparameters.

Keeping every tunable value in one dataclass means the training script, the
evaluation script and the serving layer all agree on image size, class order
and normalisation constants. That consistency matters: a model trained with
one normalisation but served with another silently produces garbage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = two levels up from this file (src/damagescan/config.py).
ROOT = Path(__file__).resolve().parents[2]

# Standard directory layout. Everything heavy (data, models) is git-ignored.
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # untouched downloaded dataset
PROCESSED_DIR = DATA_DIR / "processed"  # train/ val/ test/ with class subfolders
MODELS_DIR = ROOT / "models"         # checkpoints (.pt) and exported .onnx
REPORTS_DIR = ROOT / "reports"       # confusion matrices, Grad-CAM overlays
METRICS_PATH = ROOT / "metrics.json"

# Class names. torchvision's ImageFolder assigns integer labels by sorting the
# class subfolder names alphabetically, so the authoritative order is
# {"damaged": 0, "intact": 1}. We mirror that order here and, crucially, save
# the real order inside every checkpoint so serving never has to guess.
# "damaged" is the positive class we care about: in a sorting facility, failing
# to flag a damaged parcel (a false negative) is the costly error, so we report
# precision/recall for the "damaged" class by name, not by a hard-coded index.
CLASS_NAMES: tuple[str, ...] = ("damaged", "intact")
POSITIVE_CLASS = "damaged"

# ImageNet statistics. We use pretrained backbones, so inputs must be
# normalised exactly as those backbones were trained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """Hyperparameters for a single training run.

    Values are deliberately conservative so a run finishes quickly on a laptop
    GPU. Every field can be overridden from the command line in ``train.py``.
    """

    arch: str = "resnet18"          # "resnet18" or "mobilenet_v3_small"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3                # learning rate for the new classifier head
    weight_decay: float = 1e-4
    patience: int = 4               # early-stopping patience (epochs w/o val gain)
    num_workers: int = 2            # DataLoader worker processes
    seed: int = 42
    # Device is resolved at runtime; "auto" picks CUDA when available.
    device: str = "auto"
    class_names: tuple[str, ...] = field(default_factory=lambda: CLASS_NAMES)


def resolve_device(device: str = "auto") -> str:
    """Return a concrete torch device string.

    Imported lazily so that non-torch code (e.g. the synthetic data generator)
    does not pay the cost of importing torch.
    """
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable (secrets/paths live in .env, never code)."""
    return os.environ.get(name, default)
