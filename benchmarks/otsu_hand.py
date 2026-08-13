"""
Otsu thresholding refined with HAND and slope masking.

The spec is explicit about why this baseline matters: it "shows how much
of the gain is just physical priors, not learning" -- intellectually
honest before crediting any of it to a trained model. Raw Otsu (see
benchmarks/otsu.py) floods the map with false positives wherever
something dark-but-not-water shows up: radar shadow behind terrain, dry
smooth surfaces, and so on. This baseline takes the same raw Otsu
candidate mask and drops any pixel that physically can't be flood water --
too high above the nearest drainage channel, or sitting on too steep a
slope -- using nothing but terrain, no learning involved.
"""

import numpy as np

from benchmarks.otsu import otsu_water_mask

# Neither of these is a physical constant -- they're common starting
# points from the HAND-based flood mapping literature, not values derived
# from this project's own data. A pixel more than 15m above the nearest
# drainage channel, or sitting on more than a 5 degree slope, is treated
# as physically implausible as flood water here.
DEFAULT_HAND_THRESHOLD_M = 15.0
DEFAULT_SLOPE_THRESHOLD_DEG = 5.0


def otsu_hand_water_mask(
    vv_db: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    hand_threshold: float = DEFAULT_HAND_THRESHOLD_M,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD_DEG,
) -> np.ndarray:
    """
    Refine a raw Otsu water mask by dropping any candidate pixel that's
    too high above the nearest drainage channel or on too steep a slope
    to plausibly be flood water. NaN HAND/slope values (e.g. the DEM
    border artifact documented in data/hand.py) conservatively drop the
    pixel from the water class, the same way a NaN VV pixel does in the
    raw Otsu mask.
    """
    if not (vv_db.shape == hand.shape == slope.shape):
        raise ValueError(
            f"shape mismatch: vv_db {vv_db.shape}, hand {hand.shape}, slope {slope.shape}"
        )

    candidate_water = otsu_water_mask(vv_db)
    plausible_terrain = (hand <= hand_threshold) & (slope <= slope_threshold)
    return candidate_water & plausible_terrain
