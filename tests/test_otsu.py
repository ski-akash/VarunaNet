"""Tests for Otsu thresholding, using a synthetic bimodal VV backscatter chip."""

import numpy as np
import pytest

from benchmarks.otsu import otsu_water_mask


def make_bimodal_vv(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    A chip that's clearly two populations: the left half is dark (water-like
    backscatter), the right half is bright (land-like backscatter), each
    with a little noise so it isn't a perfectly flat step function.
    """
    rng = np.random.default_rng(seed=0)
    vv_db = np.empty((size, size), dtype=np.float32)
    vv_db[:, : size // 2] = rng.normal(loc=-20.0, scale=0.5, size=(size, size // 2))
    vv_db[:, size // 2 :] = rng.normal(loc=-8.0, scale=0.5, size=(size, size // 2))

    true_water = np.zeros((size, size), dtype=bool)
    true_water[:, : size // 2] = True
    return vv_db, true_water


def test_otsu_separates_dark_water_from_bright_land():
    vv_db, true_water = make_bimodal_vv()

    predicted_water = otsu_water_mask(vv_db)

    # Not pixel-perfect (there's noise near the boundary), but the vast
    # majority of pixels should land on the correct side of the threshold.
    agreement = (predicted_water == true_water).mean()
    assert agreement > 0.99


def test_otsu_rejects_non_2d_input():
    vv_db = np.zeros((2, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        otsu_water_mask(vv_db)


def test_otsu_handles_nan_edge_pixels():
    # Real Sentinel-1 chips can have NaN at scene-edge no-data regions.
    vv_db, true_water = make_bimodal_vv()
    vv_db[0, 0] = np.nan
    vv_db[-1, -1] = np.nan

    predicted_water = otsu_water_mask(vv_db)

    assert not predicted_water[0, 0]  # NaN pixels default to "not water"
    assert not predicted_water[-1, -1]
    agreement = (predicted_water == true_water).mean()
    assert agreement > 0.99


def test_otsu_rejects_all_nan_input():
    vv_db = np.full((8, 8), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        otsu_water_mask(vv_db)
