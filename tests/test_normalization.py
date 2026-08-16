"""
Tests for computing, saving, loading, and applying normalization stats,
using small synthetic tensors instead of real data.
"""

import numpy as np
import pytest

from data.contract import CHIP_SIZE, NUM_CHANNELS
from data.normalization import (
    NormalizationStats,
    apply_normalization,
    compute_normalization_stats,
)


def make_tensor(channel_offset: float) -> np.ndarray:
    # Each channel gets a different constant offset, so per-channel mean is
    # easy to reason about by hand when checking the computed stats.
    tensor = np.zeros((NUM_CHANNELS, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    for c in range(NUM_CHANNELS):
        tensor[c] = c + channel_offset
    return tensor


def test_compute_stats_matches_known_mean():
    tensors = [make_tensor(0.0), make_tensor(2.0)]  # per-channel values: c and c+2
    stats = compute_normalization_stats(tensors)

    expected_mean = [c + 1.0 for c in range(NUM_CHANNELS)]  # average of c and c+2
    assert np.allclose(stats.mean, expected_mean)


def test_compute_stats_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        compute_normalization_stats([])


def test_compute_stats_ignores_nan_instead_of_propagating_it():
    # Regression test for a real bug found wiring up the real dataset:
    # np.mean/np.std propagate a single NaN into a NaN result for the
    # entire channel, but real chips carry NaN by design (every chip's
    # HAND channel has NaN border pixels -- see data/hand.py), so plain
    # mean/std would have silently produced NaN stats the first time this
    # ran against real data.
    clean = make_tensor(0.0)
    with_nan = make_tensor(0.0).copy()
    with_nan[0, 0, 0] = np.nan  # one NaN pixel in channel 0

    stats = compute_normalization_stats([clean, with_nan])

    assert np.isfinite(stats.mean).all()
    assert np.isfinite(stats.std).all()
    # With only one NaN pixel out of many, channel 0's mean should still
    # land essentially at its expected value (0.0), not be corrupted.
    assert abs(stats.mean[0] - 0.0) < 1e-3


def test_save_and_load_round_trip(tmp_path):
    stats = compute_normalization_stats([make_tensor(0.0), make_tensor(2.0)])
    path = tmp_path / "norm_stats.json"

    stats.save(path)
    loaded = NormalizationStats.load(path)

    assert loaded == stats


def test_apply_normalization_centers_data():
    tensors = [make_tensor(0.0), make_tensor(2.0)]
    stats = compute_normalization_stats(tensors)

    normalized = apply_normalization(tensors[0], stats)

    # Every pixel in a channel started at the same constant value, so after
    # subtracting that channel's mean and dividing by its std, every pixel
    # in that channel should land at the same normalized value too.
    for c in range(NUM_CHANNELS):
        assert np.allclose(normalized[c], normalized[c, 0, 0])


def test_apply_normalization_rejects_channel_mismatch():
    stats = NormalizationStats(channels=("only_one_channel",), mean=(0.0,), std=(1.0,))
    tensor = make_tensor(0.0)

    with pytest.raises(ValueError, match="data contract"):
        apply_normalization(tensor, stats)
