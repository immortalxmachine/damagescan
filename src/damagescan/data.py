"""Datasets, augmentation and the inference preprocessing used at serve time.

The training and evaluation transforms live here so that the exact same
resize/normalise pipeline is reused everywhere, including inside the FastAPI
and Gradio apps. One source of truth prevents train/serve skew.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    """Return the transform pipeline for training or evaluation.

    Training applies light augmentation (random crop, flips, small rotations,
    colour jitter) so the model sees varied views and overfits less. Evaluation
    is deterministic: resize then centre-crop, so results are reproducible.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    # Evaluation / inference: resize the short side then centre-crop to square.
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),  # 256 for a 224 crop
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_dataloaders(
    processed_dir: Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create train/val/test DataLoaders from ``processed_dir``.

    ``processed_dir`` must contain ``train/``, ``val/`` and ``test/`` folders,
    each holding one subfolder per class. ``ImageFolder`` infers labels from the
    subfolder names in sorted order, which matches our CLASS_NAMES ordering
    (``damaged`` < ``intact`` alphabetically — see note below).
    """
    train_ds = ImageFolder(processed_dir / "train", transform=build_transforms(True, image_size))
    val_ds = ImageFolder(processed_dir / "val", transform=build_transforms(False, image_size))
    test_ds = ImageFolder(processed_dir / "test", transform=build_transforms(False, image_size))

    # ImageFolder sorts class folders alphabetically: {"damaged":0, "intact":1}.
    # We expose whatever order it chose so downstream code never hard-codes it.
    class_names = train_ds.classes

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader, class_names


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Turn a PIL image into a normalised (1, 3, H, W) batch tensor.

    This is the single preprocessing entry point for serving. It mirrors the
    evaluation transforms exactly so a photo scored by the API is treated the
    same way it was during evaluation.
    """
    tfm = build_transforms(train=False, image_size=image_size)
    tensor = tfm(image.convert("RGB"))
    return tensor.unsqueeze(0)  # add the batch dimension
