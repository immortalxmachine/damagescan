"""Grad-CAM explainability.

Grad-CAM answers "where did the network look?" by weighting the final
convolutional feature maps by how much each one influenced the score for a
class, then overlaying the result as a heatmap. It is implemented here from
first principles with forward/backward hooks so the mechanism is visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import MODELS_DIR, REPORTS_DIR, resolve_device
from .data import preprocess_image
from .model import get_gradcam_target_layer, load_checkpoint


class GradCAM:
    """Compute a Grad-CAM heatmap for one image and one target class."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None  # feature maps A
        self.gradients: torch.Tensor | None = None     # dScore/dA
        # Forward hook stores the layer's output; backward hook stores the
        # gradient flowing into that output during back-prop.
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1] for ``input_tensor`` (1,3,H,W)."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        # Back-propagate only the chosen class score.
        score = logits[0, class_index]
        score.backward()

        # alpha_k = global-average-pooled gradient for feature map k.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        # Weighted sum of feature maps, then ReLU (we only want positive signal).
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)

        # Upsample to the input resolution and normalise to [0, 1].
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_and_save(
    image: Image.Image, heatmap: np.ndarray, title: str, out_path: Path, image_size: int = 224
) -> None:
    """Blend the heatmap over the resized image and save a labelled figure."""
    base = image.convert("RGB").resize((image_size, image_size))
    heat = cm.jet(heatmap)[:, :, :3]  # apply the 'jet' colormap, drop alpha
    blended = (0.55 * np.asarray(base) / 255.0 + 0.45 * heat)
    blended = np.clip(blended, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(base)
    axes[0].set_title("input")
    axes[1].imshow(blended)
    axes[1].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(checkpoint: Path, image_paths: list[Path], out_dir: Path = REPORTS_DIR) -> list[Path]:
    """Generate Grad-CAM overlays for each image and return the saved paths."""
    device = resolve_device("auto")
    model, arch, class_names = load_checkpoint(checkpoint, device=device)
    gradcam = GradCAM(model, get_gradcam_target_layer(model, arch))

    saved: list[Path] = []
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path)
        x = preprocess_image(image).to(device)
        x.requires_grad_(True)
        heatmap = gradcam(x)
        pred_index = int(model(x).argmax(dim=1).item())
        title = f"Grad-CAM: {class_names[pred_index]}"
        out_path = out_dir / f"gradcam_{arch}_{i}_{class_names[pred_index]}.png"
        overlay_and_save(image, heatmap, title, out_path)
        saved.append(out_path)
        print(f"saved {out_path}")
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Grad-CAM overlays.")
    p.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "resnet18.pt")
    p.add_argument("--images", type=Path, nargs="+", required=True, help="image files")
    p.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    args = p.parse_args()
    run(args.checkpoint, args.images, args.out_dir)


if __name__ == "__main__":
    main()
