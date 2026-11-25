"""Evaluate a trained checkpoint on the held-out test set.

Produces, for one architecture:
  * accuracy, and precision / recall / F1 for the "damaged" class,
  * a confusion-matrix PNG saved under reports/,
  * an entry merged into the top-level metrics.json.

All numbers come from running the model on real test data — nothing here is
hand-written, so the README can quote metrics.json directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so it works headless / in CI
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import (
    METRICS_PATH,
    POSITIVE_CLASS,
    PROCESSED_DIR,
    REPORTS_DIR,
    resolve_device,
)
from .data import make_dataloaders
from .model import load_checkpoint


@torch.no_grad()
def predict_test_set(model, test_loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_labels, predicted_labels) over the whole test loader."""
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in test_loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
    return np.array(y_true), np.array(y_pred)


def _write_metrics(arch: str, payload: dict) -> None:
    """Merge one architecture's results into metrics.json (keyed by arch)."""
    data = {}
    if METRICS_PATH.exists():
        data = json.loads(METRICS_PATH.read_text())
    data[arch] = payload
    METRICS_PATH.write_text(json.dumps(data, indent=2))


def evaluate(
    checkpoint: Path,
    data_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    batch_size: int = 32,
    device: str = "auto",
) -> dict:
    """Evaluate ``checkpoint`` on the test split and record the results."""
    device = resolve_device(device)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model, arch, class_names = load_checkpoint(checkpoint, device=device)
    _train, _val, test_loader, loader_classes = make_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0
    )
    # Sanity check: the checkpoint's class order must match the test folders'.
    assert list(class_names) == list(loader_classes), (
        f"class order mismatch: checkpoint {class_names} vs data {loader_classes}"
    )

    y_true, y_pred = predict_test_set(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    pos_index = class_names.index(POSITIVE_CLASS)
    # Binary precision/recall/F1 for the "damaged" class (the one that matters).
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[pos_index], average="binary", pos_label=pos_index, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Save the confusion matrix as an image for the report / README.
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{arch} — confusion matrix")
    cm_path = reports_dir / f"confusion_matrix_{arch}.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    payload = {
        "accuracy": round(float(acc), 4),
        "precision_damaged": round(float(precision), 4),
        "recall_damaged": round(float(recall), 4),
        "f1_damaged": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "n_test": int(len(y_true)),
        "confusion_matrix_image": str(cm_path.relative_to(METRICS_PATH.parent)),
    }
    _write_metrics(arch, payload)

    print(f"[{arch}] acc {acc:.3f} | damaged P {precision:.3f} R {recall:.3f} F1 {f1:.3f}")
    print(f"[{arch}] confusion matrix -> {cm_path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a DamageScan checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    evaluate(args.checkpoint, data_dir=args.data_dir, device=args.device)


if __name__ == "__main__":
    main()
