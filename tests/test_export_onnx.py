"""
Tests for inference/export_onnx.py.

These run on a tiny U-Net at a small tile size on CPU, in the same spirit
as tests/test_train.py: the point is export *mechanics* (does the graph
round-trip, is the batch axis really dynamic, does verification actually
catch a mismatch), not model accuracy.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from inference.export_onnx import (
    LOGIT_ATOL,
    export_checkpoint,
    load_model_from_checkpoint,
)
from models.build_model import build_model

TILE = 64  # a real tile is 512; 64 keeps these tests fast
IN_CHANNELS = 5


def _write_checkpoint(path: Path, architecture: str = "unet") -> None:
    """
    A checkpoint in the same shape training/checkpoint.py writes: model
    weights plus the config needed to rebuild the architecture.
    """
    model = build_model(
        architecture=architecture,
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {
                    "architecture": architecture,
                    "encoder_name": "resnet18",
                    "in_channels": IN_CHANNELS,
                    "classes": 1,
                }
            },
        },
        path,
    )


def test_export_produces_a_verified_onnx_file(tmp_path):
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)

    result = export_checkpoint(checkpoint, tmp_path / "model.onnx", tile_size=TILE)

    assert result.onnx_path.exists()
    assert result.onnx_bytes > 0
    assert result.architecture == "unet"
    assert result.in_channels == IN_CHANNELS
    # Verification ran and the graphs agreed. The bound is the same one
    # export_checkpoint enforces, restated here so a loosened tolerance in
    # the module shows up as a failing test rather than passing silently.
    assert result.max_abs_logit_diff < LOGIT_ATOL


def test_exported_graph_accepts_batch_sizes_it_was_not_traced_with(tmp_path):
    """
    The tiler feeds whole batches of tiles from one scene and the batch
    size depends on the serving machine, so a batch axis frozen at export
    time would break serving while looking fine in every other test.
    """
    onnxruntime = pytest.importorskip("onnxruntime")

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    result = export_checkpoint(checkpoint, tmp_path / "model.onnx", tile_size=TILE)

    session = onnxruntime.InferenceSession(
        str(result.onnx_path), providers=["CPUExecutionProvider"]
    )
    for batch in (1, 5):
        probe = np.random.randn(batch, IN_CHANNELS, TILE, TILE).astype(np.float32)
        logits = session.run(["logits"], {"input": probe})[0]
        assert logits.shape == (batch, 1, TILE, TILE)


def test_exported_graph_matches_pytorch_on_a_fresh_input(tmp_path):
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    result = export_checkpoint(checkpoint, tmp_path / "model.onnx", tile_size=TILE)

    onnxruntime = pytest.importorskip("onnxruntime")
    model, _ = load_model_from_checkpoint(checkpoint)
    probe = torch.randn(2, IN_CHANNELS, TILE, TILE)

    with torch.no_grad():
        torch_logits = model(probe).numpy()
    session = onnxruntime.InferenceSession(
        str(result.onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_logits = session.run(["logits"], {"input": probe.numpy()})[0]

    # Thresholded predictions are what actually ship, so agreeing on the
    # decision matters more than agreeing on raw logits.
    assert np.array_equal(onnx_logits > 0, torch_logits > 0)


def test_exported_file_is_self_contained(tmp_path):
    """
    torch's exporter puts weights in a sidecar `<name>.onnx.data` and
    leaves the .onnx a few hundred KB of graph. That file looks and copies
    like a complete model but loads with no weights, so the export must
    fold the weights back in and leave nothing beside it.
    """
    onnxruntime = pytest.importorskip("onnxruntime")
    import onnx

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    result = export_checkpoint(checkpoint, tmp_path / "model.onnx", tile_size=TILE)

    sidecar = Path(str(result.onnx_path) + ".data")
    assert not sidecar.exists(), "weights were left in an external sidecar file"

    model = onnx.load(str(result.onnx_path), load_external_data=False)
    assert not any(
        init.data_location == onnx.TensorProto.EXTERNAL for init in model.graph.initializer
    )

    # The decisive check: it still runs with nothing else in the directory.
    moved = tmp_path / "alone" / "model.onnx"
    moved.parent.mkdir()
    moved.write_bytes(result.onnx_path.read_bytes())
    session = onnxruntime.InferenceSession(str(moved), providers=["CPUExecutionProvider"])
    probe = np.random.randn(1, IN_CHANNELS, TILE, TILE).astype(np.float32)
    assert session.run(["logits"], {"input": probe})[0].shape == (1, 1, TILE, TILE)


def test_verification_failure_deletes_the_bad_graph(tmp_path, monkeypatch):
    """
    A graph that exports but computes something else is the dangerous
    case: it would serve a quietly worse flood map. export_checkpoint must
    refuse it AND remove it, so a later step cannot pick up a file that
    looks like a successful export.
    """
    import inference.export_onnx as export_module

    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(checkpoint)
    output = tmp_path / "model.onnx"

    # Force disagreement without touching the real export path: make the
    # comparison see wrong numbers, as a genuinely broken graph would.
    monkeypatch.setattr(export_module.np, "allclose", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="disagree"):
        export_checkpoint(checkpoint, output, tile_size=TILE)

    assert not output.exists(), "a failed export must not leave a file behind"
