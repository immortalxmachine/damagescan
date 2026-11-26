"""Training loop with early stopping and a learning-rate scheduler.

Run from the command line, e.g.::

    python -m damagescan.train --arch resnet18 --epochs 15

The loop is written out explicitly (forward, loss, backward, step) rather than
hidden behind a high-level framework, so every operation is visible and
explainable.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import MODELS_DIR, PROCESSED_DIR, TrainConfig, resolve_device
from .data import make_dataloaders
from .model import SUPPORTED_ARCHS, build_model, freeze_backbone, save_checkpoint


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch, and make cuDNN deterministic.

    Without the cuDNN flags, GPU convolutions pick fast but non-deterministic
    algorithms, so two runs with the same seed can diverge — on a small dataset
    that is the difference between a model that learns and one that collapses to
    a single class. Deterministic mode trades a little speed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """Run one pass over ``loader``.

    When ``optimizer`` is given we train (gradients on); otherwise we evaluate
    (gradients off). Returns (mean_loss, accuracy, macro_f1).
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    # torch.set_grad_enabled lets one function serve both train and eval.
    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)           # forward pass
            loss = criterion(logits, labels)  # cross-entropy

            if is_train:
                optimizer.zero_grad()         # clear old gradients
                loss.backward()               # back-propagate
                optimizer.step()              # update weights

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)      # highest-scoring class
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    mean_loss = running_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return mean_loss, acc, f1


def train(cfg: TrainConfig, data_dir: Path = PROCESSED_DIR, out_dir: Path = MODELS_DIR) -> dict:
    """Train one model end to end and return its training history."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Arch: {cfg.arch}")

    train_loader, val_loader, _test_loader, class_names = make_dataloaders(
        data_dir, cfg.image_size, cfg.batch_size, cfg.num_workers
    )
    print(f"Classes (label order): {class_names}")

    model = build_model(cfg.arch, num_classes=len(class_names), pretrained=True).to(device)
    if cfg.freeze_backbone:
        freeze_backbone(model, cfg.arch)
    criterion = nn.CrossEntropyLoss()
    # Optimise only the parameters that require gradients (all of them for a full
    # fine-tune, or just the head when the backbone is frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_trainable:,} (freeze_backbone={cfg.freeze_backbone})")
    optimizer = Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Halve the LR when validation loss plateaus, so late training fine-tunes.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_f1 = -1.0
    epochs_without_gain = 0
    history: list[dict] = []
    ckpt_path = out_dir / f"{cfg.arch}.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"| train loss {tr_loss:.3f} f1 {tr_f1:.3f} "
            f"| val loss {val_loss:.3f} f1 {val_f1:.3f} "
            f"| {time.time() - t0:.1f}s"
        )

        # Checkpoint the best model by validation macro-F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_gain = 0
            save_checkpoint(model, cfg.arch, tuple(class_names), ckpt_path)
            print(f"  -> new best (val f1 {val_f1:.3f}) saved to {ckpt_path.name}")
        else:
            # Early stopping: bail out once val F1 stops improving.
            epochs_without_gain += 1
            if epochs_without_gain >= cfg.patience:
                print(f"  -> early stopping after {cfg.patience} epochs without gain")
                break

    history_path = out_dir / f"{cfg.arch}_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Best val macro-F1: {best_val_f1:.3f} | history -> {history_path.name}")
    return {"best_val_f1": best_val_f1, "history": history, "checkpoint": str(ckpt_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Train a DamageScan classifier.")
    p.add_argument("--arch", choices=SUPPORTED_ARCHS, default=cfg.arch)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--patience", type=int, default=cfg.patience)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--freeze-backbone", action="store_true", help="train only the classifier head")
    p.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--device", default=cfg.device)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = TrainConfig(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
    )
    train(cfg, data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
