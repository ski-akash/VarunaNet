"""
Otsu thresholding: the classical, non-ML baseline for SAR water detection.

The spec calls this out specifically: "the standard operational SAR
water-detection method. The 'do you even need ML' baseline. Must be
included." Every fancier model in this project has to actually beat this,
or the fancier model isn't earning its complexity.

The physical reasoning it relies on: water is dark (low backscatter) in
SAR because it acts as a specular reflector, scattering the radar pulse
away from the sensor instead of back to it. Land is comparatively bright.
Otsu's method finds the pixel value that best splits a bimodal histogram
into two classes -- here, dark water pixels vs. bright land pixels --
automatically, using no labels and no training at all.

Thresholds VH, not VV, and denoises before thresholding -- both confirmed
against real evidence, not assumed. The original per-chip-VV version of
this baseline scored far below the published Sen1Floods11 numbers (mean
IoU 0.211-0.224 here vs. a published Otsu-VH IoU of 0.2850-0.3862,
Bonafilia et al., CVPRW 2020); switching the thresholded band from VV to
VH and adding speckle denoising before thresholding closed that gap:
scoring this project's own VH+denoise reconstruction got to mean IoU
0.30, and independently, scoring the *dataset's own shipped*
S1OtsuLabelHand precomputed Otsu labels (via this project's own metrics
pipeline, on the exact same official test split) landed at mean IoU
0.309 -- both lines of evidence agreeing confirms this isn't
overfitting to one number.
"""

import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu

# A 7x7 median filter was the best-performing kernel size tested against
# real data among {3, 5, 7, 9, 11} (mean/median IoU peaked at 7, then
# declined -- too large a kernel starts blurring away real, smaller water
# bodies, not just speckle noise).
DEFAULT_SMOOTHING_SIZE = 7


def smooth_backscatter(band_db: np.ndarray, size: int = DEFAULT_SMOOTHING_SIZE) -> np.ndarray:
    """
    Median-filters a backscatter (dB) chip to reduce SAR speckle noise
    before thresholding -- raw per-pixel backscatter is noisy enough that
    Otsu's histogram split (and the resulting mask) is visibly worse
    without this; see this module's docstring for the real numbers.

    NaN-aware: NaN pixels (scene-edge no-data) are filled with the chip's
    own finite mean before filtering -- a neutral value, not 0 dB, which
    would otherwise bias the filter toward "water" near a no-data
    border -- then the original NaN mask is restored afterward, so NaN
    pixels stay NaN rather than picking up a filtered neighbor's value.
    """
    finite = band_db[np.isfinite(band_db)]
    fill_value = float(finite.mean()) if finite.size > 0 else 0.0
    filled = np.where(np.isfinite(band_db), band_db, fill_value)
    smoothed = median_filter(filled, size=size)
    return np.where(np.isfinite(band_db), smoothed, np.nan)


def compute_otsu_threshold(band_db: np.ndarray) -> float:
    """
    Computes Otsu's threshold from an array of backscatter (dB) values.
    Not restricted to a single chip's 2D shape: callers that want a
    threshold shared across multiple chips (see benchmarks/evaluate.py's
    compute_per_event_otsu_thresholds) pass in pixels pooled from every
    chip in a flood event, since Otsu's method needs a real amount of
    bimodal signal -- dark water vs. bright land -- to find a good split,
    and a single chip often doesn't have enough of both classes in it to
    do that well on its own.
    """
    finite_values = band_db[np.isfinite(band_db)]
    if finite_values.size == 0:
        raise ValueError("no finite backscatter values to threshold")
    return float(threshold_otsu(finite_values))


def otsu_water_mask(band_db: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """
    Threshold a backscatter (dB) chip into a binary water mask. Pixels
    darker than `threshold` are classified as water. In practice this
    project calls it with the VH band (see this module's docstring for
    why), but the function itself is band-agnostic -- it just thresholds
    whatever 2D array it's given.

    If `threshold` is None, it's computed from this one chip's own
    histogram (the original per-chip behavior, still what every direct
    caller of this function uses). Pass a precomputed threshold instead
    when scoring within a shared context that has more signal to draw
    on -- see compute_otsu_threshold's docstring for why that matters,
    and benchmarks/evaluate.py's compute_per_event_otsu_thresholds for
    where this project actually does it.
    """
    if band_db.ndim != 2:
        raise ValueError(f"expected a 2D backscatter array, got shape {band_db.shape}")

    if threshold is None:
        threshold = compute_otsu_threshold(band_db)

    # Real SAR chips can have NaN at scene-edge no-data regions -- a known
    # Sentinel-1 artifact, not something wrong with this function.
    # Comparing NaN < threshold naturally evaluates to False, so those
    # pixels default to "not water", which is harmless since they're
    # excluded from evaluation via the label's ignore index anyway. This
    # also means a chip that's entirely NaN (e.g. real test chip
    # Paraguay_34417) naturally predicts "no water anywhere" when given a
    # precomputed threshold, with no separate fallback path needed --
    # only computing a threshold from nothing (the threshold=None path)
    # can actually raise.
    return band_db < threshold
