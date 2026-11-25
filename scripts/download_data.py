"""Download the parcel dataset into data/raw/.

Primary path (real project): the Kaggle dataset of intact vs damaged packages.
You need a Kaggle API token:
  1. kaggle.com -> your avatar -> Settings -> API -> "Create New Token"
  2. Save the downloaded kaggle.json to  %USERPROFILE%\\.kaggle\\kaggle.json
     (Windows) or ~/.kaggle/kaggle.json (macOS/Linux).
  3. Find the dataset on Kaggle and copy its slug "owner/dataset-name" from the
     URL (kaggle.com/datasets/<owner>/<dataset-name>).

Then run, for example::

    python scripts/download_data.py --source kaggle --dataset <owner>/<name>

Fallback (no token / CI / quick demo): generate synthetic images instead::

    python scripts/download_data.py --source synthetic --n-per-class 80

After downloading, arrange data/raw so there is one subfolder per class
("intact" and "damaged"), then run scripts/make_splits.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src/ package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from damagescan.config import RAW_DIR  # noqa: E402

# The dataset referenced by the project brief: synthetic (Blender-rendered) RGB
# images of intact vs damaged packages, two views per package. License: GPL-2.
DEFAULT_KAGGLE_DATASET = "christianvorhemus/industrial-quality-control-of-packages"


def download_kaggle(dataset: str, out_dir: Path) -> None:
    """Download and unzip a Kaggle dataset using the official kaggle client."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The 'kaggle' package is not installed. Run: pip install kaggle"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json
    print(f"Downloading {dataset} -> {out_dir} ...")
    api.dataset_download_files(dataset, path=str(out_dir), unzip=True)
    print("Done. Inspect the folder and ensure it has per-class subfolders.")
    print("Tip: if classes are nested, point make_splits.py --src at the right folder.")


def generate_synthetic(out_dir: Path, n_per_class: int) -> None:
    """Generate a clearly-labelled synthetic dataset (fallback / tests)."""
    from damagescan.synthetic import generate_dataset

    total = generate_dataset(out_dir, n_per_class=n_per_class)
    print(f"Generated {total} SYNTHETIC images under {out_dir} (label this in the README).")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", choices=["kaggle", "synthetic"], default="kaggle")
    p.add_argument(
        "--dataset",
        default=DEFAULT_KAGGLE_DATASET,
        help=f"Kaggle slug 'owner/dataset-name' (default: {DEFAULT_KAGGLE_DATASET})",
    )
    p.add_argument("--out-dir", type=Path, default=RAW_DIR)
    p.add_argument("--n-per-class", type=int, default=80, help="synthetic images per class")
    args = p.parse_args()

    if args.source == "kaggle":
        if not args.dataset:
            p.error("--dataset 'owner/name' is required when --source kaggle")
        download_kaggle(args.dataset, args.out_dir)
    else:
        generate_synthetic(args.out_dir, args.n_per_class)


if __name__ == "__main__":
    main()
