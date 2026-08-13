"""
Tests for HAND computation, using a small synthetic valley DEM with a
known drainage line, instead of a real terrain tile.
"""

import numpy as np
import pytest

from data.hand import compute_hand

# Chosen to sit above the accumulation this synthetic DEM's far ridge
# builds up as it drains sideways into the valley (see make_valley_dem),
# so only the true drainage line gets marked as drainage. A lower
# threshold picks up that ridge too, since it forms its own long-but-
# shallow tributary -- a good reminder that this parameter is genuinely
# terrain-dependent, not a fixed constant.
ACCUMULATION_THRESHOLD = 15


def make_valley_dem(size: int = 9) -> np.ndarray:
    """
    A DEM shaped like a valley: elevation rises 5m per pixel moving away
    from a drainage line down column 2, plus a gentle overall downhill
    slope (varying by row) so flow has somewhere to go.
    """
    dem = np.zeros((size, size), dtype=np.float64)
    for col in range(size):
        dem[:, col] = abs(col - 2) * 5.0 + 10.0
    for row in range(size):
        dem[row, :] += (size - row) * 0.1
    return dem


def test_hand_is_zero_at_the_drainage_line():
    dem = make_valley_dem()

    hand = compute_hand(dem, pixel_size_m=10.0, accumulation_threshold=ACCUMULATION_THRESHOLD)

    # Interior rows only -- the outer border is a documented NaN edge effect.
    assert np.allclose(hand[1:-1, 2], 0.0)


def test_hand_matches_elevation_above_the_drainage_line():
    dem = make_valley_dem()

    hand = compute_hand(dem, pixel_size_m=10.0, accumulation_threshold=ACCUMULATION_THRESHOLD)

    # Every column drains laterally into column 2 at its own row before
    # flowing onward, so HAND should exactly match each column's height
    # above column 2 in that same row: abs(col - 2) * 5m.
    row = 4
    for col in range(1, 8):
        expected = abs(col - 2) * 5.0
        assert hand[row, col] == pytest.approx(expected)


def test_hand_border_is_nan():
    dem = make_valley_dem()

    hand = compute_hand(dem, pixel_size_m=10.0, accumulation_threshold=ACCUMULATION_THRESHOLD)

    assert np.isnan(hand[0, :]).all()
    assert np.isnan(hand[-1, :]).all()
    assert np.isnan(hand[:, 0]).all()


def test_hand_output_is_float32():
    dem = make_valley_dem()

    hand = compute_hand(dem, pixel_size_m=10.0, accumulation_threshold=ACCUMULATION_THRESHOLD)

    assert hand.dtype == np.float32
    assert hand.shape == dem.shape


def test_rejects_non_2d_input():
    dem = np.zeros((2, 5, 5), dtype=np.float64)

    with pytest.raises(ValueError, match="2D"):
        compute_hand(dem, pixel_size_m=10.0)
