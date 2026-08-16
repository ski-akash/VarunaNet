"""
Tests for the data contract, run against small synthetic tensors rather
than real satellite data. The point of these tests is to catch a shape or
dtype mistake immediately, on a laptop, instead of after a training job has
already been queued on the cluster.
"""

import numpy as np
import pytest

from data.contract import (
    CHIP_SIZE,
    LABEL_IGNORE,
    LABEL_NON_WATER,
    LABEL_WATER,
    NUM_CHANNELS,
    validate_input_tensor,
    validate_label_tensor,
)


def make_valid_input() -> np.ndarray:
    return np.random.randn(NUM_CHANNELS, CHIP_SIZE, CHIP_SIZE).astype(np.float32)


def make_valid_label() -> np.ndarray:
    return np.random.choice(
        [LABEL_NON_WATER, LABEL_WATER, LABEL_IGNORE],
        size=(CHIP_SIZE, CHIP_SIZE),
    ).astype(np.int64)


def test_valid_input_tensor_passes():
    validate_input_tensor(make_valid_input())


def test_input_tensor_wrong_channel_count_rejected():
    bad = np.random.randn(NUM_CHANNELS + 1, CHIP_SIZE, CHIP_SIZE).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_input_tensor(bad)


def test_input_tensor_wrong_dtype_rejected():
    bad = np.random.randn(NUM_CHANNELS, CHIP_SIZE, CHIP_SIZE).astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        validate_input_tensor(bad)


def test_input_tensor_with_nan_rejected():
    # Real chips carry NaN by design (e.g. HAND's border pixels -- see
    # data/hand.py), so it must be resolved to a real value before a
    # tensor reaches this point, not after -- this is the check that
    # catches it if that step is ever skipped.
    bad = make_valid_input()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_input_tensor(bad)


def test_valid_label_tensor_passes():
    validate_label_tensor(make_valid_label())


def test_label_tensor_wrong_shape_rejected():
    bad = np.zeros((CHIP_SIZE, CHIP_SIZE + 1), dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        validate_label_tensor(bad)


def test_label_tensor_unexpected_value_rejected():
    bad = make_valid_label()
    bad[0, 0] = 2  # only 0, 1, and 255 are valid label values
    with pytest.raises(ValueError, match="unexpected values"):
        validate_label_tensor(bad)
