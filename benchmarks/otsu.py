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
"""

import numpy as np
from skimage.filters import threshold_otsu


def otsu_water_mask(vv_db: np.ndarray) -> np.ndarray:
    """
    Threshold a VV backscatter (dB) chip into a binary water mask using
    Otsu's method. Pixels darker than the automatically chosen threshold
    are classified as water.
    """
    if vv_db.ndim != 2:
        raise ValueError(f"expected a 2D VV array, got shape {vv_db.shape}")

    finite_values = vv_db[np.isfinite(vv_db)]
    if finite_values.size == 0:
        raise ValueError("no finite VV values to threshold")

    # Real SAR chips can have NaN at scene-edge no-data regions -- a known
    # Sentinel-1 artifact, not something wrong with this function. The
    # threshold is computed from real values only; comparing NaN < threshold
    # naturally evaluates to False, so those pixels default to "not water",
    # which is harmless since they're excluded from evaluation via the
    # label's ignore index anyway.
    threshold = threshold_otsu(finite_values)
    return vv_db < threshold
