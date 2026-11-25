"""Export a trained checkpoint to ONNX and verify it.

ONNX is a framework-neutral model format. Exporting lets the serving layer run
the model with onnxruntime — a light dependency with no PyTorch required — and
makes the model portable to other runtimes or hardware.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import MODELS_DIR
from .model import load_checkpoint


def export(checkpoint: Path, out_path: Path | None = None, image_size: int = 224) -> Path:
    """Export ``checkpoint`` to ONNX and sanity-check it against PyTorch."""
    import json

    model, arch, class_names = load_checkpoint(checkpoint, device="cpu")
    out_path = out_path or (MODELS_DIR / f"{arch}.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ONNX graphs carry no label metadata, so write the class order to a sidecar
    # file the serving layer reads. This keeps label meaning attached to weights.
    classes_path = out_path.with_suffix(".classes.json")
    classes_path.write_text(json.dumps(list(class_names)))

    # A dummy input defines the graph's tensor shapes during tracing. We mark
    # the batch axis as dynamic so the exported model accepts any batch size.
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Verify: PyTorch and onnxruntime should agree on the same input.
    import onnxruntime as ort

    torch_out = model(dummy).detach().numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    assert max_diff < 1e-3, f"ONNX output diverges from PyTorch by {max_diff}"

    print(f"Exported {arch} -> {out_path} (max |Δ| vs torch = {max_diff:.2e})")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
