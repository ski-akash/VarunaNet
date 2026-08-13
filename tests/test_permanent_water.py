"""
Tests for permanent water masking and flood extent computation, using
small synthetic occurrence/detection arrays with known correct answers.
"""

import numpy as np
import pytest

from data.permanent_water import compute_flood_extent, compute_permanent_water_mask


def test_high_occurrence_pixels_are_marked_permanent():
    occurrence = np.array([[80.0, 30.0], [60.0, 10.0]])

    mask = compute_permanent_water_mask(occurrence, threshold=50.0)

    assert np.array_equal(mask, [[True, False], [True, False]])


def test_threshold_boundary_is_inclusive():
    occurrence = np.array([[50.0, 49.9]])

    mask = compute_permanent_water_mask(occurrence, threshold=50.0)

    assert np.array_equal(mask, [[True, False]])


def test_rejects_out_of_range_occurrence_values():
    occurrence = np.array([[150.0, 0.0]])

    with pytest.raises(ValueError, match="range"):
        compute_permanent_water_mask(occurrence)


def test_rejects_non_2d_occurrence():
    occurrence = np.zeros((2, 3, 3))

    with pytest.raises(ValueError, match="2D"):
        compute_permanent_water_mask(occurrence)


def test_flood_extent_excludes_permanent_water():
    # The model detected water everywhere in this 2x2 tile...
    detected_water = np.array([[True, True], [True, True]])
    # ...but half of it is a river that's there every day anyway.
    permanent_water_mask = np.array([[True, False], [False, True]])

    flood_extent = compute_flood_extent(detected_water, permanent_water_mask)

    assert np.array_equal(flood_extent, [[False, True], [True, False]])


def test_flood_extent_rejects_shape_mismatch():
    detected_water = np.zeros((2, 2), dtype=bool)
    permanent_water_mask = np.zeros((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_flood_extent(detected_water, permanent_water_mask)
