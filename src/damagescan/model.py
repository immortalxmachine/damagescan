"""Model construction via transfer learning.

We start from backbones pretrained on ImageNet and replace only the final
classification layer. The convolutional features learned on ImageNet transfer
well to parcel photos, so we need far less data and far less training time
than learning from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# Architectures this project supports. Keeping the list explicit makes the CLI
# validate user input and documents exactly what we compared in the report.
SUPPORTED_ARCHS = ("resnet18", "mobilenet_v3_small")


def build_model(arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a backbone with a fresh classification head.

    Parameters
    ----------
    arch:
        One of :data:`SUPPORTED_ARCHS`.
    num_classes:
        Output units (2 for intact/damaged).
    pretrained:
        Load ImageNet weights. Set ``False`` in tests for a tiny untrained model.
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        # ResNet's classifier is a single Linear named ``fc``. Swap it for one
        # sized to our problem; its weights start random and are trained.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # MobileNetV3's head is a small Sequential; the last Linear is index -1.
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {SUPPORTED_ARCHS}")

    return model


def get_gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the last convolutional block to hook for Grad-CAM.

    Grad-CAM needs the activations and gradients of the final conv feature map,
    because that is where spatial information still exists before global
    pooling collapses it.
    """
    if arch == "resnet18":
        # layer4 is the last residual stage; its output is 7x7x512 for 224px in.
        return model.layer4[-1]
    if arch == "mobilenet_v3_small":
        # features[-1] is the final conv+activation block.
        return model.features[-1]
    raise ValueError(f"Unknown arch '{arch}'. Supported: {SUPPORTED_ARCHS}")


def save_checkpoint(model: nn.Module, arch: str, class_names: tuple[str, ...], path) -> None:
    """Persist weights plus the metadata needed to rebuild and interpret them."""
    torch.save(
        {
            "arch": arch,
            "class_names": list(class_names),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, str, list[str]]:
    """Rebuild a model from a checkpoint saved by :func:`save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    class_names = ckpt["class_names"]
    model = build_model(arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, arch, class_names
