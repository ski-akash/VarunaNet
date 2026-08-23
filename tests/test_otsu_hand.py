"""
Tests for Otsu+HAND refinement, using a synthetic scene with a "radar
shadow" patch: dark in the backscatter band like water, but on steep,
high-HAND terrain that should never actually flood. otsu_hand_water_mask
is band-agnostic (see benchmarks/otsu_hand.py) -- these tests don't need
to care that benchmarks/evaluate.py actually feeds it VH in practice.
"""

import numpy as np
import pytest

from benchmarks.otsu_hand import otsu_hand_water_mask, plausible_terrain_mask


def make_scene(size: int = 64):
    rng = np.random.default_rng(seed=1)
    band_db = np.empty((size, size), dtype=np.float32)
    band_db[:, : size // 2] = rng.normal(loc=-20.0, scale=0.5, size=(size, size // 2))
    band_db[:, size // 2 :] = rng.normal(loc=-8.0, scale=0.5, size=(size, size // 2))

    hand = np.full((size, size), 2.0, dtype=np.float32)
    slope = np.full((size, size), 1.0, dtype=np.float32)

    # A patch that's dark (Otsu alone would call it water) but sits high
    # above drainage on steep terrain -- classic radar shadow, not water.
    shadow_region = np.s_[10:20, 10:20]
    hand[shadow_region] = 50.0
    slope[shadow_region] = 20.0

    return band_db, hand, slope, shadow_region


def test_otsu_hand_excludes_radar_shadow():
    band_db, hand, slope, shadow_region = make_scene()

    predicted_water = otsu_hand_water_mask(band_db, hand, slope)

    assert not predicted_water[shadow_region].any()


def test_otsu_hand_keeps_plausible_water_elsewhere():
    band_db, hand, slope, shadow_region = make_scene()

    predicted_water = otsu_hand_water_mask(band_db, hand, slope)
    raw_water = predicted_water.copy()
    raw_water[shadow_region] = True  # ignore the deliberately-excluded patch

    # Everywhere else on the dark (left) side, plausible-terrain water
    # should still be detected -- the terrain mask shouldn't suppress it
    # everywhere, only where the terrain actually disqualifies it.
    left_half = np.s_[:, :32]
    assert predicted_water[left_half].mean() > 0.8


def test_otsu_hand_rejects_shape_mismatch():
    band_db = np.zeros((8, 8), dtype=np.float32)
    hand = np.zeros((8, 9), dtype=np.float32)
    slope = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        otsu_hand_water_mask(band_db, hand, slope)


def test_otsu_hand_water_mask_passes_precomputed_threshold_through_to_otsu():
    band_db, hand, slope, _shadow_region = make_scene()

    # threshold=0.0 forces every pixel to read as candidate water in the
    # underlying otsu_water_mask (see test_otsu.py's equivalent case) --
    # the terrain mask here is uniform aside from the shadow patch, so
    # the whole non-shadow region should come back water.
    predicted_water = otsu_hand_water_mask(band_db, hand, slope, otsu_threshold=0.0)

    non_shadow = predicted_water.copy()
    assert non_shadow[0, 0]  # top-left corner: outside the shadow patch, terrain is plausible


def test_nan_terrain_does_not_reject_a_water_pixel():
    """
    Regression test for the bug that made this baseline score WORSE than plain
    Otsu. `hand <= threshold` evaluates to False for NaN, so the original
    implementation silently rejected every pixel where HAND could not be
    computed -- and data/hand.py leaves a NaN border on every real chip.

    Measured on the real test split, that removed 192,740 true water pixels
    (9.75% of all true positives) against only 68,217 false ones: chip borders
    preferentially cut through rivers, which is exactly where HAND's flow
    routing fails. NaN terrain must mean "unknown, abstain", not "reject".
    """
    band_db = np.full((4, 4), -25.0)  # dark everywhere -> Otsu says water
    hand = np.full((4, 4), np.nan)
    slope = np.full((4, 4), np.nan)

    mask = otsu_hand_water_mask(band_db, hand, slope, otsu_threshold=-20.0)

    assert mask.all(), "NaN terrain must not veto Otsu's own water decision"


def test_plausible_terrain_mask_still_rejects_on_known_bad_terrain():
    """The NaN fix must not disable the filter where terrain IS known."""
    hand = np.array([[1.0, 100.0], [1.0, 1.0]])
    slope = np.array([[1.0, 1.0], [80.0, 1.0]])

    mask = plausible_terrain_mask(hand, slope, hand_threshold=5.0, slope_threshold=15.0)

    assert mask[0, 0]  # low HAND, flat -> plausible
    assert not mask[0, 1]  # HAND 100m -> implausible
    assert not mask[1, 0]  # slope 80deg -> implausible
    assert mask[1, 1]


def test_plausible_terrain_mask_mixes_nan_and_known_values():
    """A NaN in one layer must not suppress a real rejection in the other."""
    hand = np.array([[np.nan, np.nan]])
    slope = np.array([[1.0, 80.0]])

    mask = plausible_terrain_mask(hand, slope, hand_threshold=5.0, slope_threshold=15.0)

    assert mask[0, 0]  # HAND unknown, slope fine -> keep
    assert not mask[0, 1]  # HAND unknown, but slope is definitively too steep
