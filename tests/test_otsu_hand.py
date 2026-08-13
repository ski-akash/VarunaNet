"""
Tests for Otsu+HAND refinement, using a synthetic scene with a "radar
shadow" patch: dark in VV like water, but on steep, high-HAND terrain
that should never actually flood.
"""

import numpy as np
import pytest

from benchmarks.otsu_hand import otsu_hand_water_mask


def make_scene(size: int = 64):
    rng = np.random.default_rng(seed=1)
    vv_db = np.empty((size, size), dtype=np.float32)
    vv_db[:, : size // 2] = rng.normal(loc=-20.0, scale=0.5, size=(size, size // 2))
    vv_db[:, size // 2 :] = rng.normal(loc=-8.0, scale=0.5, size=(size, size // 2))

    hand = np.full((size, size), 2.0, dtype=np.float32)
    slope = np.full((size, size), 1.0, dtype=np.float32)

    # A patch that's dark in VV (Otsu alone would call it water) but sits
    # high above drainage on steep terrain -- classic radar shadow, not water.
    shadow_region = np.s_[10:20, 10:20]
    hand[shadow_region] = 50.0
    slope[shadow_region] = 20.0

    return vv_db, hand, slope, shadow_region


def test_otsu_hand_excludes_radar_shadow():
    vv_db, hand, slope, shadow_region = make_scene()

    predicted_water = otsu_hand_water_mask(vv_db, hand, slope)

    assert not predicted_water[shadow_region].any()


def test_otsu_hand_keeps_plausible_water_elsewhere():
    vv_db, hand, slope, shadow_region = make_scene()

    predicted_water = otsu_hand_water_mask(vv_db, hand, slope)
    raw_water = predicted_water.copy()
    raw_water[shadow_region] = True  # ignore the deliberately-excluded patch

    # Everywhere else on the dark (left) side, plausible-terrain water
    # should still be detected -- the terrain mask shouldn't suppress it
    # everywhere, only where the terrain actually disqualifies it.
    left_half = np.s_[:, :32]
    assert predicted_water[left_half].mean() > 0.8


def test_otsu_hand_rejects_shape_mismatch():
    vv_db = np.zeros((8, 8), dtype=np.float32)
    hand = np.zeros((8, 9), dtype=np.float32)
    slope = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        otsu_hand_water_mask(vv_db, hand, slope)
