"""
Tests for Otsu thresholding, using a synthetic bimodal backscatter chip.
otsu_water_mask/compute_otsu_threshold are band-agnostic (they just
threshold whatever 2D/1D array they're given) -- benchmarks/evaluate.py
is what decides to pass them the VH band specifically, see that module.
"""

import numpy as np
import pytest

from benchmarks.otsu import compute_otsu_threshold, otsu_water_mask, smooth_backscatter


def make_bimodal_band(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    A chip that's clearly two populations: the left half is dark (water-like
    backscatter), the right half is bright (land-like backscatter), each
    with a little noise so it isn't a perfectly flat step function.
    """
    rng = np.random.default_rng(seed=0)
    band_db = np.empty((size, size), dtype=np.float32)
    band_db[:, : size // 2] = rng.normal(loc=-20.0, scale=0.5, size=(size, size // 2))
    band_db[:, size // 2 :] = rng.normal(loc=-8.0, scale=0.5, size=(size, size // 2))

    true_water = np.zeros((size, size), dtype=bool)
    true_water[:, : size // 2] = True
    return band_db, true_water


def test_otsu_separates_dark_water_from_bright_land():
    band_db, true_water = make_bimodal_band()

    predicted_water = otsu_water_mask(band_db)

    # Not pixel-perfect (there's noise near the boundary), but the vast
    # majority of pixels should land on the correct side of the threshold.
    agreement = (predicted_water == true_water).mean()
    assert agreement > 0.99


def test_otsu_rejects_non_2d_input():
    band_db = np.zeros((2, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        otsu_water_mask(band_db)


def test_otsu_handles_nan_edge_pixels():
    # Real Sentinel-1 chips can have NaN at scene-edge no-data regions.
    band_db, true_water = make_bimodal_band()
    band_db[0, 0] = np.nan
    band_db[-1, -1] = np.nan

    predicted_water = otsu_water_mask(band_db)

    assert not predicted_water[0, 0]  # NaN pixels default to "not water"
    assert not predicted_water[-1, -1]
    agreement = (predicted_water == true_water).mean()
    assert agreement > 0.99


def test_otsu_rejects_all_nan_input():
    band_db = np.full((8, 8), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        otsu_water_mask(band_db)


def test_otsu_water_mask_uses_a_precomputed_threshold_when_given():
    band_db, _true_water = make_bimodal_band()
    # A threshold well above every pixel's value forces the whole chip to
    # read as water -- if this were ignored in favor of computing the
    # chip's own threshold, the result would look like the untouched
    # bimodal split instead.
    predicted_water = otsu_water_mask(band_db, threshold=0.0)

    assert predicted_water.all()


def test_otsu_water_mask_with_precomputed_threshold_does_not_raise_on_all_nan():
    # The no-finite-values ValueError only applies to computing a
    # threshold from nothing (threshold=None) -- a precomputed threshold
    # skips that computation entirely, so an all-NaN chip just predicts
    # no water anywhere instead of raising. This is exactly what lets
    # benchmarks/evaluate.py's make_otsu_predict handle a real all-NaN
    # test chip (Paraguay_34417) without a separate try/except fallback.
    band_db = np.full((8, 8), np.nan, dtype=np.float32)

    predicted_water = otsu_water_mask(band_db, threshold=-15.0)

    assert not predicted_water.any()


def test_compute_otsu_threshold_matches_otsu_water_masks_own_default():
    band_db, _true_water = make_bimodal_band()

    threshold = compute_otsu_threshold(band_db)

    assert np.array_equal(otsu_water_mask(band_db), band_db < threshold)


def test_compute_otsu_threshold_rejects_all_nan_input():
    band_db = np.full((8, 8), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        compute_otsu_threshold(band_db)


def test_compute_otsu_threshold_accepts_pooled_1d_pixels():
    # compute_per_event_otsu_thresholds (benchmarks/evaluate.py) pools
    # finite pixels from every chip in an event into one flat 1D array,
    # not a single chip's 2D shape -- compute_otsu_threshold (unlike
    # otsu_water_mask) has to accept that.
    band_db, _true_water = make_bimodal_band()
    pooled = band_db[np.isfinite(band_db)].ravel()

    threshold = compute_otsu_threshold(pooled)

    assert np.isfinite(threshold)


def test_smooth_backscatter_reduces_pixel_to_pixel_speckle_noise():
    rng = np.random.default_rng(seed=2)
    clean = np.full((32, 32), -15.0, dtype=np.float32)
    speckled = clean + rng.normal(scale=3.0, size=clean.shape).astype(np.float32)

    smoothed = smooth_backscatter(speckled, size=7)

    # A median filter over a uniform-in-expectation field should land much
    # closer to the true constant value than the raw speckled input did.
    assert np.abs(smoothed - clean).mean() < np.abs(speckled - clean).mean()


def test_smooth_backscatter_keeps_nan_pixels_nan():
    band_db, _true_water = make_bimodal_band()
    band_db[0, 0] = np.nan

    smoothed = smooth_backscatter(band_db, size=5)

    assert np.isnan(smoothed[0, 0])
    assert np.isfinite(smoothed[1, 1])


def test_smooth_backscatter_does_not_let_nan_leak_into_neighbors():
    # A NaN-filled neighbor value would otherwise corrupt the neighboring
    # finite pixels' filtered result -- smooth_backscatter fills NaN with
    # the chip's own finite mean before filtering specifically to avoid this.
    band_db, _true_water = make_bimodal_band()
    band_db[0, 0] = np.nan

    smoothed = smooth_backscatter(band_db, size=3)

    assert np.isfinite(smoothed[0, 1])
    assert np.isfinite(smoothed[1, 0])
