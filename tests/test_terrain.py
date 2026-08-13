"""Tests for slope computation, using small synthetic DEM patches with known answers."""

import numpy as np
import pytest

from data.terrain import compute_slope


def test_flat_dem_has_zero_slope():
    dem = np.full((5, 5), fill_value=100.0, dtype=np.float32)

    slope = compute_slope(dem, pixel_size_m=10.0)

    assert np.allclose(slope, 0.0)


def test_constant_ramp_has_expected_slope_angle():
    # Elevation rises 10m for every 10m pixel moved in the x direction:
    # a rise/run of 1, i.e. a 45 degree slope, at every point on the ramp.
    width = 5
    row = np.arange(width, dtype=np.float32) * 10.0
    dem = np.tile(row, (width, 1))

    slope = compute_slope(dem, pixel_size_m=10.0)

    assert np.allclose(slope, 45.0, atol=1e-4)


def test_steeper_ramp_has_larger_slope_angle():
    shallow_dem = np.tile(np.arange(5, dtype=np.float32) * 5.0, (5, 1))
    steep_dem = np.tile(np.arange(5, dtype=np.float32) * 20.0, (5, 1))

    shallow_slope = compute_slope(shallow_dem, pixel_size_m=10.0)
    steep_slope = compute_slope(steep_dem, pixel_size_m=10.0)

    assert steep_slope[0, 0] > shallow_slope[0, 0]


def test_rejects_non_2d_input():
    dem = np.zeros((2, 5, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        compute_slope(dem, pixel_size_m=10.0)
