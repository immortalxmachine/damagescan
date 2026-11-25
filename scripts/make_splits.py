"""Create a stratified train/val/test split from class-labelled raw images.

Input:  a source directory with one subfolder per class, e.g.
            data/raw/intact/*.jpg
            data/raw/damaged/*.jpg
Output: data/processed/{train,val,test}/<class>/*.jpg

"Stratified" means each split keeps the same class proportions as the whole
dataset, so the test accuracy is not skewed by, say, all the damaged examples
landing in training. The split is seeded, so it is reproducible.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from damagescan.config import PROCESSED_DIR, RAW_DIR  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dirs(src: Path) -> list[Path]:
    """Return subfolders of ``src`` that contain at least one image."""
    class_dirs = []
    for child in sorted(src.iterdir()):
        if child.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in child.rglob("*")):
            class_dirs.append(child)
    return class_dirs


def split_class(files: list[Path], ratios: tuple[float, float, float], rng: random.Random):
    """Shuffle and split one class's files into (train, val, test) lists."""
    files = files[:]
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def make_splits(
    src: Path = RAW_DIR,
    dst: Path = PROCESSED_DIR,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    name_filter: str | None = None,
) -> dict:
    """Copy images from ``src`` into stratified splits under ``dst``.

    ``name_filter`` keeps only images whose path (relative to the class folder)
    contains the given substring. For the Kaggle packages dataset we pass
    ``"top"`` to use only the top-view image of each package: that gives exactly
    one image per package, so no two views of the same package can land in
    different splits (which would leak information from train into test).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    class_dirs = find_class_dirs(src)
    if not class_dirs:
        raise SystemExit(f"No class subfolders with images found under {src}")

    rng = random.Random(seed)
    # Start from a clean destination so re-runs are deterministic.
    if dst.exists():
        shutil.rmtree(dst)

    counts: dict[str, dict[str, int]] = {}
    for class_dir in class_dirs:
        cls = class_dir.name
        files = [f for f in class_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
        if name_filter:
            files = [f for f in files if name_filter in str(f.relative_to(class_dir))]
        train, val, test = split_class(files, ratios, rng)
        counts[cls] = {"train": len(train), "val": len(val), "test": len(test), "total": len(files)}
        for split_name, split_files in (("train", train), ("val", val), ("test", test)):
            out = dst / split_name / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out / f.name)

    print(f"Classes: {[d.name for d in class_dirs]}")
    for cls, c in counts.items():
        print(
            f"  {cls:10s} total {c['total']:5d} -> "
            f"train {c['train']} val {c['val']} test {c['test']}"
        )
    return counts


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, default=RAW_DIR)
    p.add_argument("--dst", type=Path, default=PROCESSED_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--name-filter",
        default=None,
        help="keep only images whose relative path contains this substring (e.g. 'top')",
    )
    args = p.parse_args()
    make_splits(args.src, args.dst, seed=args.seed, name_filter=args.name_filter)


if __name__ == "__main__":
    main()
