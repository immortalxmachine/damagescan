"""Procedural synthetic parcel images.

This module draws cartoon parcels so the pipeline can run end-to-end without any
download — used by the unit tests (tiny, fast) and as an optional fallback for
``download_data.py`` when no Kaggle token is available. Images are deliberately
simple: a cardboard box on a background, where "damaged" boxes carry dark
tears/dents and holes. They are clearly synthetic and must be labelled as such
in any report; the real project trains on the Kaggle dataset.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_CARDBOARD = (193, 154, 107)
_BACKGROUNDS = [(230, 230, 230), (200, 205, 210), (180, 180, 175)]


def make_synthetic_image(label: str, size: int = 224, seed: int | None = None) -> Image.Image:
    """Draw one synthetic parcel image for ``label`` ("intact" or "damaged")."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), rng.choice(_BACKGROUNDS))
    draw = ImageDraw.Draw(img)

    # Box position and size vary so the model cannot rely on fixed geometry.
    margin = rng.randint(20, 45)
    x0, y0 = margin, margin
    x1, y1 = size - margin, size - margin
    box_colour = tuple(max(0, c + rng.randint(-20, 20)) for c in _CARDBOARD)
    draw.rectangle([x0, y0, x1, y1], fill=box_colour, outline=(90, 70, 50), width=3)

    # Packing tape: a couple of lighter horizontal/vertical strips.
    tape = (215, 205, 180)
    ty = rng.randint(y0 + 20, y1 - 20)
    draw.rectangle([x0, ty - 8, x1, ty + 8], fill=tape)

    if label == "damaged":
        # Dark jagged tears, dents and holes concentrated on the box surface.
        for _ in range(rng.randint(3, 6)):
            cx = rng.randint(x0 + 10, x1 - 10)
            cy = rng.randint(y0 + 10, y1 - 10)
            n_pts = rng.randint(4, 7)
            pts = [(cx + rng.randint(-30, 30), cy + rng.randint(-30, 30)) for _ in range(n_pts)]
            dark = (rng.randint(10, 50),) * 3
            draw.polygon(pts, fill=dark)
        # A crushed corner: overdraw one corner darker and distorted.
        corner = rng.choice([(x0, y0), (x1, y0), (x0, y1), (x1, y1)])
        draw.ellipse(
            [corner[0] - 25, corner[1] - 25, corner[0] + 25, corner[1] + 25],
            fill=(60, 45, 35),
        )

    # Light sensor noise for realism.
    arr = np.asarray(img).astype(np.int16)
    arr += rng.choice([4, 6, 8]) * np.random.default_rng(seed).integers(-3, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_dataset(out_dir: Path, n_per_class: int = 60, size: int = 224, seed: int = 0) -> int:
    """Write ``n_per_class`` images for each class under ``out_dir/<class>/``.

    Returns the total number of images written.
    """
    total = 0
    for label in ("damaged", "intact"):
        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img_seed = seed * 100000 + (hash(label) % 1000) + i
            img = make_synthetic_image(label, size=size, seed=img_seed)
            img.save(class_dir / f"{label}_{i:04d}.jpg", quality=90)
            total += 1
    return total
