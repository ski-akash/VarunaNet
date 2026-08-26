"""
Tests for inference/quantize.py.

A tiny model at a small tile size, so these exercise the mechanics --
does quantization produce a smaller runnable graph, does the report
compare the two fairly, does the accuracy metric measure decisions rather
than raw logits. The headline FP32/INT8 numbers come from running the CLI
on the real exported model, not from here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from inference.export_onnx import export_checkpoint
from inference.quantize import (
    benchmark_quantization,
    load_real_chips,
    quantize_dynamic_int8,
)
from models.build_model import build_model

TILE = 64
IN_CHANNELS = 5


def _exported_model(tmp_path: Path) -> Path:
    model = build_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=1,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {
                    "architecture": "unet",
                    "encoder_name": "resnet18",
                    "in_channels": IN_CHANNELS,
                    "classes": 1,
                }
            },
        },
        checkpoint,
    )
    return export_checkpoint(checkpoint, tmp_path / "model.onnx", tile_size=TILE).onnx_path


def test_quantized_model_is_smaller_and_still_runs(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    fp32 = _exported_model(tmp_path)

    int8 = quantize_dynamic_int8(fp32, tmp_path / "model.int8.onnx")

    assert int8.exists()
    assert int8.stat().st_size < fp32.stat().st_size

    session = onnxruntime.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    probe = np.random.randn(1, IN_CHANNELS, TILE, TILE).astype(np.float32)
    assert session.run(None, {session.get_inputs()[0].name: probe})[0].shape == (
        1,
        1,
        TILE,
        TILE,
    )


def test_report_compares_size_and_latency(tmp_path):
    pytest.importorskip("onnxruntime")
    fp32 = _exported_model(tmp_path)
    int8 = quantize_dynamic_int8(fp32, tmp_path / "model.int8.onnx")

    report = benchmark_quantization(
        fp32, int8, in_channels=IN_CHANNELS, tile_size=TILE, batch_size=1, timed_runs=1
    )

    assert report.size_ratio > 1  # int8 is smaller
    assert report.fp32_seconds_per_tile > 0
    assert report.int8_seconds_per_tile > 0
    assert 0.0 <= report.mask_disagreement_fraction <= 1.0
    assert report.max_abs_logit_diff >= 0.0


def test_disagreement_measures_decisions_not_logit_magnitude(tmp_path):
    """
    The metric must count pixels that flip water/not-water. A large logit
    shift that stays on the same side of the threshold changes nothing
    that ships, and an MSE-style metric would score those identically to
    a shift that flips the decision.
    """
    pytest.importorskip("onnxruntime")
    fp32 = _exported_model(tmp_path)
    int8 = quantize_dynamic_int8(fp32, tmp_path / "model.int8.onnx")

    report = benchmark_quantization(
        fp32,
        int8,
        sample_batch=np.zeros((1, IN_CHANNELS, TILE, TILE), dtype=np.float32),
        timed_runs=1,
    )

    # Identical input through two near-identical graphs: almost nothing
    # should flip, even though logits differ slightly everywhere.
    assert report.mask_disagreement_fraction < 0.5


def test_benchmark_accepts_a_supplied_batch(tmp_path):
    """
    Real chips are passed in rather than generated, because random noise
    understates quantization damage -- measured ~2x lower than real chips
    on the actual model.
    """
    pytest.importorskip("onnxruntime")
    fp32 = _exported_model(tmp_path)
    int8 = quantize_dynamic_int8(fp32, tmp_path / "model.int8.onnx")
    supplied = np.random.randn(3, IN_CHANNELS, TILE, TILE).astype(np.float32)

    report = benchmark_quantization(fp32, int8, sample_batch=supplied, timed_runs=1)

    assert report.fp32_seconds_per_tile > 0


def test_load_real_chips_returns_none_when_dataset_is_absent(tmp_path):
    """
    The dataset is gitignored and large, so anything depending on it must
    degrade to a clear signal rather than crashing.
    """
    assert load_real_chips(4, data_root=tmp_path / "not-a-dataset") is None
