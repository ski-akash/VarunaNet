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
