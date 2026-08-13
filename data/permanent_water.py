"""
Permanent water masking from the JRC Global Surface Water dataset.

SAR detects "water" -- but a river is water every single day of the year.
Without subtracting out where water sits permanently, every scene of the
Brahmaputra would read as "flooded", even on an ordinary dry-season day.
The JRC Global Surface Water "occurrence" layer -- the percentage of
satellite observations between 1984 and 2021 that found water at each
pixel -- is what makes that subtraction possible, and it's what actually
turns "the model found water here" into the project's real deliverable:
flood extent, not just water extent.
"""

import numpy as np

# JRC occurrence values are a percentage (0-100): what fraction of
# historical satellite observations found water at this pixel. A pixel
# needs to be wet most of the time, not just occasionally, to count as
# "permanent" rather than seasonal -- this default treats "wet more than
# half the time on record" as permanent water.
DEFAULT_OCCURRENCE_THRESHOLD = 50.0


def compute_permanent_water_mask(
    occurrence: np.ndarray, threshold: float = DEFAULT_OCCURRENCE_THRESHOLD
) -> np.ndarray:
    """
    Threshold a JRC occurrence layer (0-100, percent of historical
    observations that found water) into a binary permanent water mask.
    """
    if occurrence.ndim != 2:
        raise ValueError(f"expected a 2D occurrence array, got shape {occurrence.shape}")
    if np.any((occurrence < 0) | (occurrence > 100)):
        raise ValueError("occurrence values must be in the range [0, 100]")

    return occurrence >= threshold


def compute_flood_extent(
    detected_water: np.ndarray, permanent_water_mask: np.ndarray
) -> np.ndarray:
    """
    Flood extent = water the model detected, minus water that's there on
    any ordinary day anyway. This is the actual product deliverable --
    "flood extent", not "water extent" -- and the entire reason the
    permanent water mask exists (see module docstring).
    """
    if detected_water.shape != permanent_water_mask.shape:
        raise ValueError(
            f"shape mismatch: detected_water {detected_water.shape} vs "
            f"permanent_water_mask {permanent_water_mask.shape}"
        )

    return detected_water & ~permanent_water_mask
