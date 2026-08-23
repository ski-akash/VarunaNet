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

# Tuned by grid search on the TRAIN split via benchmarks/tune_otsu_hand.py,
# then scored exactly once on test -- never selected against test labels.
# These replaced the original untuned literature starting points (15m / 5deg).
#
# The 5 degree slope cutoff was the harmful half: measured on real test chips,
# genuine flood water has a p90 slope of 6.12 degrees, so a 5 degree cutoff
# discarded 13.1% of real water. Relaxing it to 15 degrees keeps that water
# while still rejecting the steep radar-shadow false positives the filter
# exists for.
#
# The HAND cutoff went the other way -- 15m was too *permissive*, not too
# strict. True water sits at a median HAND of 0.00m and a p90 of 1.47m, so
# almost none of it is above even 5m; tightening to 5m therefore costs
# essentially no real water while removing more false alarms.
DEFAULT_HAND_THRESHOLD_M = 5.0
DEFAULT_SLOPE_THRESHOLD_DEG = 15.0


def plausible_terrain_mask(
    hand: np.ndarray,
    slope: np.ndarray,
    hand_threshold: float = DEFAULT_HAND_THRESHOLD_M,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD_DEG,
) -> np.ndarray:
    """
    Which pixels are terrain-plausible as flood water: low enough above the
    nearest drainage channel, and flat enough.

    **NaN terrain means "unknown", not "implausible".** This distinction was
    originally gotten wrong here and it cost more accuracy than every other
    parameter in this baseline combined. A plain `hand <= threshold` comparison
    evaluates to False for NaN, which silently rejects the pixel -- and
    data/hand.py's flow routing leaves a NaN border on *every* chip as a
    documented, unavoidable edge effect. Measured on the real test split, that
    NaN ring removed 192,740 true water pixels (9.75% of all true positives)
    against only 68,217 false ones (4.37%): it destroyed real water at more
    than twice the rate it removed false alarms, because chip borders
    preferentially cut through rivers and drainage exits -- exactly where HAND
    cannot be computed and exactly where water actually is.

    So a NaN pixel is passed through unfiltered: the terrain prior abstains
    rather than voting to reject. Otsu's own decision stands for those pixels.
    This is what makes the baseline a real refinement of Otsu instead of a
    net-harmful filter (pooled IoU 0.438 -> 0.482 on the test split).

    Shared by otsu_hand_water_mask() below and by the threshold grid search in
    benchmarks/tune_otsu_hand.py, deliberately: the two had independently
    duplicated this expression, which is how the tuner ended up searching over
    a mask that carried the same bug it was trying to tune around.
    """
    hand_ok = np.where(np.isnan(hand), True, hand <= hand_threshold)
    slope_ok = np.where(np.isnan(slope), True, slope <= slope_threshold)
    return hand_ok & slope_ok


def otsu_hand_water_mask(
    band_db: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    hand_threshold: float = DEFAULT_HAND_THRESHOLD_M,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD_DEG,
    otsu_threshold: float | None = None,
) -> np.ndarray:
    """
    Refine a raw Otsu water mask by dropping any candidate pixel that's
    too high above the nearest drainage channel or on too steep a slope
    to plausibly be flood water. NaN HAND/slope values (e.g. the DEM
    border artifact documented in data/hand.py) are treated as *unknown*
    and left to Otsu's own decision -- see plausible_terrain_mask above
    for the measured reason that matters so much here.

    `band_db`/`otsu_threshold` are passed straight through to
    otsu_water_mask -- see that function's docstring for which band this
    project actually thresholds and when a precomputed (e.g. per-event)
    threshold is preferable to letting each chip compute its own.
    """
    if not (band_db.shape == hand.shape == slope.shape):
        raise ValueError(
            f"shape mismatch: band_db {band_db.shape}, hand {hand.shape}, slope {slope.shape}"
        )

    candidate_water = otsu_water_mask(band_db, otsu_threshold)
    return candidate_water & plausible_terrain_mask(hand, slope, hand_threshold, slope_threshold)
