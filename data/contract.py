"""
The data contract: the exact tensor shapes, channel order, and label values
every part of this project has to agree on.

The spec calls this out specifically because a silent mismatch here (e.g. a
model trained on channels in one order being served data in a different
order) is one of the most common ways an ML project quietly produces wrong
results without ever throwing an error. Putting these values in one place,
and validating against them, means a mistake shows up as a loud crash
instead of a subtly wrong flood map.
"""

import numpy as np

# Every input tensor has these 5 channels, in this exact order:
#   VV_db            - VV-polarized radar backscatter, in decibels
#   VH_db             - VH-polarized radar backscatter, in decibels
#   VV_VH_ratio       - ratio of the two polarizations (helps separate water from look-alikes)
#   slope             - terrain slope, derived from a DEM (water doesn't sit on steep slopes)
#   HAND              - Height Above Nearest Drainage (strong prior on where water can pool)
CHANNELS = ("VV_db", "VH_db", "VV_VH_ratio", "slope", "HAND")
NUM_CHANNELS = len(CHANNELS)

# Every chip (a training/inference tile) is a fixed 512x512 square.
CHIP_SIZE = 512

# Label values, per pixel:
LABEL_NON_WATER = 0
LABEL_WATER = 1
LABEL_IGNORE = 255  # no-data pixels: excluded from the loss, not treated as "non-water"


def validate_input_tensor(tensor: np.ndarray) -> None:
    """
    Raise ValueError if `tensor` doesn't match the input contract:
    shape [NUM_CHANNELS, CHIP_SIZE, CHIP_SIZE], dtype float32, no NaN.

    The no-NaN check matters because NaN is expected upstream (real chips
    carry it -- e.g. HAND's flow-routing leaves every chip's border as
    NaN by design, see data/hand.py) but must never reach this point: a
    NaN anywhere in a convolution's receptive field poisons that whole
    output region, and that propagates layer by layer until the model's
    entire forward pass is NaN. Anything producing a final input tensor
    (e.g. training/sen1floods11_dataset.py) is responsible for replacing
    NaN with a defined value before this check, not after.
    """
    expected_shape = (NUM_CHANNELS, CHIP_SIZE, CHIP_SIZE)
    if tensor.shape != expected_shape:
        raise ValueError(f"input tensor has shape {tensor.shape}, expected {expected_shape}")
    if tensor.dtype != np.float32:
        raise ValueError(f"input tensor has dtype {tensor.dtype}, expected float32")
    if np.isnan(tensor).any():
        raise ValueError("input tensor contains NaN values -- must be resolved before this point")


def validate_label_tensor(tensor: np.ndarray) -> None:
    """
    Raise ValueError if `tensor` doesn't match the label contract:
    shape [CHIP_SIZE, CHIP_SIZE], dtype int64, values only in
    {LABEL_NON_WATER, LABEL_WATER, LABEL_IGNORE}.
    """
    expected_shape = (CHIP_SIZE, CHIP_SIZE)
    if tensor.shape != expected_shape:
        raise ValueError(f"label tensor has shape {tensor.shape}, expected {expected_shape}")
    if tensor.dtype != np.int64:
        raise ValueError(f"label tensor has dtype {tensor.dtype}, expected int64")

    allowed_values = {LABEL_NON_WATER, LABEL_WATER, LABEL_IGNORE}
    present_values = set(np.unique(tensor).tolist())
    unexpected = present_values - allowed_values
    if unexpected:
        raise ValueError(f"label tensor contains unexpected values: {unexpected}")
