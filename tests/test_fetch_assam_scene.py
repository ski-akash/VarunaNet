"""
Tests only the pure logic in data/fetch_assam_scene.py -- everything else
in that module needs a live Google Earth Engine session and real network
access, which this project's regular test suite deliberately doesn't
depend on (same reasoning as every other real-data script here: verified
live, by hand, once real credentials existed -- see VarunaNet_Spec.md's
session log -- not re-mocked into a fake here).
"""

import numpy as np

from data.fetch_assam_scene import sanity_check_against_sen1floods11_stats


def test_sanity_check_reports_real_medians_against_the_stored_stats():
    vv = np.full((4, 4), -10.0, dtype=np.float32)
    vh = np.full((4, 4), -17.0, dtype=np.float32)
    vv_vh = np.stack([vv, vh])

    result = sanity_check_against_sen1floods11_stats(vv_vh)

    assert result["vv_median"] == -10.0
    assert result["vh_median"] == -17.0
    # The comparison values come from the real, committed
    # sen1floods11_normalization_stats.json, not hardcoded here -- a
    # regression in that file should show up as a changed test value.
    assert result["sen1floods11_vv_mean"] < 0
    assert result["sen1floods11_vh_mean"] < 0


def test_sanity_check_ignores_nan_pixels():
    vv = np.array([[-10.0, np.nan], [-12.0, -8.0]], dtype=np.float32)
    vh = np.array([[-17.0, -17.0], [np.nan, -17.0]], dtype=np.float32)
    vv_vh = np.stack([vv, vh])

    result = sanity_check_against_sen1floods11_stats(vv_vh)

    assert np.isfinite(result["vv_median"])
    assert np.isfinite(result["vh_median"])
