"""
Tests for the Copernicus DEM tile naming logic, which is the one part of
data/fetch_dem.py that doesn't need real network access to verify.
Fetching real tiles is exercised manually against the real dataset instead
of in the automated suite, the same way benchmarks/otsu.py was validated
against real Sen1Floods11 chips -- pytest shouldn't depend on a third-
party bucket being reachable.
"""

from data.fetch_dem import dem_tile_url


def test_tile_url_for_northern_eastern_hemisphere():
    url = dem_tile_url(lat=9.43, lon=-1.15)
    assert "N09_00_W002_00" in url


def test_tile_url_floors_toward_the_correct_tile_not_toward_zero():
    # -1.15 is inside the W002 tile (2W-1W), not W001 -- floor(-1.15) is
    # -2, and naively truncating toward zero would give the wrong tile.
    url = dem_tile_url(lat=9.0, lon=-1.15)
    assert "W002_00" in url


def test_tile_url_for_southern_hemisphere():
    url = dem_tile_url(lat=-14.5, lon=-65.5)
    assert "S15_00_W066_00" in url


def test_tile_url_for_eastern_hemisphere():
    url = dem_tile_url(lat=26.3, lon=92.7)
    assert "N26_00_E092_00" in url


def test_tile_url_at_exact_integer_degree():
    url = dem_tile_url(lat=9.0, lon=5.0)
    assert "N09_00_E005_00" in url
