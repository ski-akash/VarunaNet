"""
Exports a trained checkpoint to ONNX, and verifies the exported graph
actually reproduces the PyTorch model's output before anyone trusts it.

Why ONNX at all (spec section 5): the serving path must not depend on the
training cluster. A checkpoint is a pickle of PyTorch tensors that needs
torch, torchvision and segmentation-models-pytorch to load -- roughly a
2GB dependency tree, and a CUDA image on top of that if it is ever to run
on GPU. ONNX Runtime is a single ~50MB wheel with no torch, which is what
lets infra/Dockerfile.serve stay slim while Dockerfile.train stays heavy.
It also gives the project a real quantization/latency story to measure
(FP32 vs FP16 vs INT8) rather than asserted.

The export is verified, not assumed. `torch.onnx.export` succeeding proves
only that a graph was written; it does not prove the graph computes the
same function. Silent divergence is the normal failure mode here -- an op
gets decomposed differently, a shape gets baked in where it should have
stayed dynamic -- and it would show up much later as a quietly worse flood
map. So `export_checkpoint` always runs both models on the same random
input and compares, and raises if they disagree.

The batch axis is dynamic because the tiler feeds whole batches of 512x512
tiles from one scene (a 25,000 x 16,000 Sentinel-1 scene is ~1,500 tiles),
and the batch size that fits depends on the machine doing the serving, not
on anything fixed at export time. Height and width are deliberately NOT
dynamic: the model is trained on a fixed tile size, the tiler's whole job
is to produce exactly that, and pinning them lets the runtime specialise.

Run directly:
    python -m inference.export_onnx --checkpoint path/to/best.pt --output model.onnx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from models.build_model import build_model

# The tile size the model is trained on (spec section 3.3's data contract).
DEFAULT_TILE_SIZE = 512

# ONNX opset. 17 is the first with native LayerNormalization, which
# SegFormer needs -- on older opsets it decomposes into a dozen primitive
# ops that are slower and harder to read when something goes wrong.
DEFAULT_OPSET = 17

# Protobuf caps a single message at 2GB, which is what forces weights into
# a sidecar file above this size. Every architecture in this project is far
# below it (the largest, U-Net++/ResNet-34, is ~100MB), so a self-contained
# file is always achievable here -- but the check is explicit rather than
# assumed, because the failure at 2GB is an opaque protobuf error.
PROTOBUF_SIZE_LIMIT_BYTES = 2 * 1024**3

# Tolerances for the PyTorch-vs-ONNX comparison. These are floating-point
# reassociation differences (a fused op summing in a different order), not
# a licence for the graph to be meaningfully different: 1e-4 on logits is
# far below what could flip a thresholded prediction, since the sigmoid
# would have to be within 1e-4 of the 0.5 cutoff for it to matter.
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-3


@dataclass
class ExportResult:
    onnx_path: Path
    checkpoint_path: Path
    architecture: str
    in_channels: int
    tile_size: int
    opset: int
    max_abs_logit_diff: float
    checkpoint_bytes: int
    onnx_bytes: int


def load_model_from_checkpoint(
    checkpoint_path: Path, device: str = "cpu"
) -> tuple[torch.nn.Module, dict]:
    """
    Rebuild the trained model from a checkpoint's own stored config.

    Mirrors training/evaluate_checkpoints.py's loader rather than going
    through training/checkpoint.py's load_checkpoint: that one also
    restores optimizer, scheduler and RNG state, none of which mean
    anything for a read-only export.

    model_cfg.get("architecture", "unet") matches the same default
    evaluate_checkpoints.py applies, for the earliest checkpoints that
    predate ModelConfig having an `architecture` field at all -- back when
    the project had exactly one architecture, so a checkpoint missing the
    key is necessarily a plain U-Net.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    model = build_model(
        architecture=model_cfg.get("architecture", "unet"),
        encoder_name=model_cfg["encoder_name"],
        # No ImageNet fetch: load_state_dict overwrites these weights on the
        # next line anyway, and the serving path must not need network access.
        encoder_weights=None,
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["config"]



def _inline_external_weights(onnx_path: Path) -> None:
    """
    Rewrite `onnx_path` so all weights live inside it, and delete the
    sidecar file the exporter may have written alongside.

    torch's exporter stores initializers in an external `<name>.onnx.data`
    file, leaving the .onnx itself a few hundred KB of pure graph. That is
    a deployment trap: the .onnx looks like a complete model, copies like
    one, and reports a plausible-looking small size, but loading it without
    its sidecar yields a model with no weights. For a 100MB model there is
    no reason to carry two coupled files into a container image, so this
    folds them back into one.

    Left alone above PROTOBUF_SIZE_LIMIT_BYTES, where a single-file model
    is not representable and the split is genuinely required.
    """
    import onnx

    model = onnx.load(str(onnx_path))  # resolves external data
    total = model.ByteSize()
    if total >= PROTOBUF_SIZE_LIMIT_BYTES:
        return

    onnx.save(model, str(onnx_path), save_as_external_data=False)
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    sidecar.unlink(missing_ok=True)


def _has_external_initializers(onnx_path: Path) -> bool:
    """True if any weight still lives outside the .onnx file."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return any(
        init.data_location == onnx.TensorProto.EXTERNAL for init in model.graph.initializer
    )


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    opset: int = DEFAULT_OPSET,
) -> ExportResult:
    """
    Export `checkpoint_path` to ONNX at `output_path`, then verify the
    exported graph matches PyTorch on a random input.

    Raises RuntimeError if the two disagree beyond LOGIT_ATOL/LOGIT_RTOL,
    and deletes the bad file rather than leaving a graph on disk that
    looks exported but computes something else.
    """
    import onnxruntime  # local import: only the export path needs it

    model, config = load_model_from_checkpoint(checkpoint_path)
    in_channels = config["model"]["in_channels"]
    architecture = config["model"].get("architecture", "unet")

    # Batch of 2, not 1: exporting with batch=1 makes it far easier for a
    # size-1 axis to get silently baked in as a constant even when it is
    # named dynamic, and the failure only appears when the tiler sends a
    # real batch.
    example = torch.randn(2, in_channels, tile_size, tile_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    _inline_external_weights(output_path)

    # Verification, on a *different* random input than the one traced, and
    # at a different batch size -- so this also proves the dynamic batch
    # axis survived rather than being folded to a constant 2.
    probe = torch.randn(3, in_channels, tile_size, tile_size)
    with torch.no_grad():
        torch_logits = model(probe).cpu().numpy()

    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    onnx_logits = session.run(["logits"], {"input": probe.numpy()})[0]

    if onnx_logits.shape != torch_logits.shape:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ONNX output shape {onnx_logits.shape} != PyTorch {torch_logits.shape}. "
            "The dynamic batch axis most likely did not survive export."
        )

    if _has_external_initializers(output_path):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{output_path} still references external weight data. Shipping it "
            "alone would produce a model with no weights."
        )

    max_abs_diff = float(np.max(np.abs(onnx_logits - torch_logits)))
    if not np.allclose(onnx_logits, torch_logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ONNX and PyTorch disagree: max abs logit difference {max_abs_diff:.3e} "
            f"exceeds atol={LOGIT_ATOL}. The exported graph was deleted rather than "
            "left on disk looking usable."
        )

    return ExportResult(
        onnx_path=output_path,
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        in_channels=in_channels,
        tile_size=tile_size,
        opset=opset,
        max_abs_logit_diff=max_abs_diff,
        checkpoint_bytes=checkpoint_path.stat().st_size,
        onnx_bytes=output_path.stat().st_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    result = export_checkpoint(
        args.checkpoint, args.output, tile_size=args.tile_size, opset=args.opset
    )
    print(
        f"exported {result.architecture} ({result.in_channels}ch, "
        f"{result.tile_size}x{result.tile_size}, opset {result.opset})\n"
        f"  {result.checkpoint_path} -> {result.onnx_path}\n"
        f"  checkpoint {result.checkpoint_bytes / 1e6:.1f} MB "
        f"-> onnx {result.onnx_bytes / 1e6:.1f} MB\n"
        f"  verified against PyTorch: max abs logit diff {result.max_abs_logit_diff:.3e}"
    )


if __name__ == "__main__":
    main()
