"""
Tests for inference/vectorize.py.

Masks are constructed by hand so the expected geometry and area are known
exactly, rather than asserting on whatever the tracer happens to produce.
"""

import numpy as np
import pytest
from affine import Affine

from inference.vectorize import (
    DEFAULT_MIN_AREA_M2,
    _pixel_area_native,
    iter_wkt,
    to_feature_collection,
    vectorize_mask,
)

# 10m pixels, north-up, origin at (0, 0) -- Sentinel-1 GRD's usual layout.
TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
PIXEL_AREA = 100.0  # m2


def _mask(height=40, width=40) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


def test_pixel_area_uses_the_full_affine_determinant():
    assert _pixel_area_native(TRANSFORM) == pytest.approx(PIXEL_AREA)
    # A rotated transform still has 100 m2 pixels; using abs(a)*abs(e)
    # would report 0 here.
    rotated = Affine(0.0, 10.0, 0.0, -10.0, 0.0, 0.0)
    assert _pixel_area_native(rotated) == pytest.approx(PIXEL_AREA)


def test_a_solid_block_becomes_one_polygon_with_the_right_area():
    mask = _mask()
    mask[5:25, 5:25] = True  # 20x20 pixels = 400 px = 40,000 m2

    result = vectorize_mask(mask, TRANSFORM, min_area_m2=0.0)

    assert len(result.polygons) == 1
    assert result.polygons[0].area_m2 == pytest.approx(40_000.0)
    assert result.total_area_hectares == pytest.approx(4.0)


def test_specks_are_dropped_and_counted():
    """
    A real SAR mask contains thousands of one- and two-pixel blobs. Left
    in they dominate the polygon count and the payload while representing
    nothing anyone would call a flood.
    """
    mask = _mask()
    mask[5:25, 5:25] = True  # a real flood, 40,000 m2
    mask[30, 30] = True  # 100 m2 speck
    mask[32, 35] = True  # another
    mask[35, 2] = True  # another

    result = vectorize_mask(mask, TRANSFORM, min_area_m2=DEFAULT_MIN_AREA_M2)

    assert len(result.polygons) == 1
    assert result.dropped_as_speck == 3
    assert result.traced_count == 4


def test_holes_are_preserved():
    """
    A dry patch inside a flooded area is real information. Dropping
    interiors would systematically overstate flood extent, which is the
    direction of error that matters most here.
    """
    mask = _mask()
    mask[5:25, 5:25] = True
    mask[12:18, 12:18] = False  # 6x6 dry patch = 3,600 m2

    result = vectorize_mask(mask, TRANSFORM, min_area_m2=0.0)

    assert len(result.polygons) == 1
    assert result.polygons[0].area_m2 == pytest.approx(40_000.0 - 3_600.0)
    assert len(result.polygons[0].geometry["coordinates"]) == 2  # exterior + 1 hole


def test_separate_regions_become_separate_polygons_sorted_by_area():
    mask = _mask(60, 60)
    mask[2:22, 2:22] = True  # 40,000 m2
    mask[30:50, 30:45] = True  # 30,000 m2

    result = vectorize_mask(mask, TRANSFORM, min_area_m2=0.0)

    assert len(result.polygons) == 2
    areas = [p.area_m2 for p in result.polygons]
    assert areas == sorted(areas, reverse=True), "largest flood should come first"
    assert result.total_area_m2 == pytest.approx(70_000.0)


def test_simplification_removes_staircase_vertices():
    """
    Traced polygons follow pixel edges, so a diagonal boundary becomes a
    staircase with a vertex every 10m. Simplifying should cut the vertex
    count substantially without moving the boundary meaningfully.
    """
    mask = _mask(60, 60)
    for row in range(2, 50):
        mask[row, 2 : row + 1] = True  # a triangle: diagonal hypotenuse

    detailed = vectorize_mask(mask, TRANSFORM, min_area_m2=0.0, simplify_pixel_fraction=0.0)
    simplified = vectorize_mask(mask, TRANSFORM, min_area_m2=0.0, simplify_pixel_fraction=0.5)

    detailed_vertices = len(detailed.polygons[0].geometry["coordinates"][0])
    simplified_vertices = len(simplified.polygons[0].geometry["coordinates"][0])

    assert simplified_vertices < detailed_vertices
    # Still the same flood: area must not move by more than a rounding of
    # the staircase itself.
    assert simplified.total_area_m2 == pytest.approx(detailed.total_area_m2, rel=0.02)


def test_an_empty_mask_produces_nothing_rather_than_failing():
    result = vectorize_mask(_mask(), TRANSFORM)

    assert result.polygons == []
    assert result.total_area_m2 == 0.0
    assert to_feature_collection(result)["features"] == []


def test_polygons_are_georeferenced_not_in_pixel_space():
    mask = _mask()
    mask[5:25, 5:25] = True

    result = vectorize_mask(mask, TRANSFORM, min_area_m2=0.0)
    coords = result.polygons[0].geometry["coordinates"][0]
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]

    # Pixel col 5 -> x=50, col 25 -> x=250; rows go negative (north-up).
    assert min(xs) == pytest.approx(50.0)
    assert max(xs) == pytest.approx(250.0)
    assert max(ys) == pytest.approx(-50.0)
    assert min(ys) == pytest.approx(-250.0)


def test_feature_collection_is_valid_geojson_with_areas():
    mask = _mask()
    mask[5:25, 5:25] = True

    collection = to_feature_collection(vectorize_mask(mask, TRANSFORM, min_area_m2=0.0))

    assert collection["type"] == "FeatureCollection"
    feature = collection["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["area_hectares"] == pytest.approx(4.0)


def test_wkt_output_is_ready_for_postgis():
    mask = _mask()
    mask[5:25, 5:25] = True

    wkt, area = next(iter_wkt(vectorize_mask(mask, TRANSFORM, min_area_m2=0.0)))

    assert wkt.startswith("POLYGON")
    assert area == pytest.approx(40_000.0)


def test_rejects_a_non_2d_mask():
    with pytest.raises(ValueError, match="2D mask"):
        vectorize_mask(np.zeros((2, 10, 10), dtype=bool), TRANSFORM)


def test_geographic_crs_areas_are_geodesic_not_square_degrees():
    """
    Sen1Floods11 chips are EPSG:4326, so the affine is in degrees. Treating
    its determinant as an area gives ~8e-09 per pixel, every polygon falls
    under any sane threshold, and a fully flooded scene silently reports
    zero. This is that regression: it was caught on a real chip that was
    77.6% water and reported 0.0 hectares.
    """
    from rasterio.crs import CRS

    # ~0.0001 degree pixels near the equator, as the real chips have.
    geo_transform = Affine(1e-4, 0.0, 105.0, 0.0, -1e-4, 11.0)
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:25, 5:25] = True  # 20x20 pixels

    result = vectorize_mask(
        mask, geo_transform, crs=CRS.from_epsg(4326), min_area_m2=DEFAULT_MIN_AREA_M2
    )

    assert len(result.polygons) == 1, "a real flood was discarded as a speck"
    # 20 x 0.0001 deg is ~222m a side at this latitude -> ~4.9 hectares.
    assert 3.0 < result.total_area_hectares < 8.0


def test_geographic_and_projected_scenes_of_the_same_ground_size_agree():
    """
    The same physical flood should report the same area whether the scene
    is stored in degrees or metres -- otherwise district-level totals would
    depend on the CRS the scene happened to arrive in.
    """
    from rasterio.crs import CRS

    mask = np.zeros((40, 40), dtype=bool)
    mask[5:25, 5:25] = True

    # 20 pixels of 1e-4 deg at the equator ~ 20 pixels of 11.1m
    geo = vectorize_mask(
        mask, Affine(1e-4, 0.0, 0.0, 0.0, -1e-4, 0.0),
        crs=CRS.from_epsg(4326), min_area_m2=0.0,
    )
    metric = vectorize_mask(
        mask, Affine(11.132, 0.0, 0.0, 0.0, -11.132, 0.0), min_area_m2=0.0
    )

    assert geo.total_area_m2 == pytest.approx(metric.total_area_m2, rel=0.02)


def test_no_crs_keeps_the_projected_interpretation():
    """
    Omitting the CRS must not silently switch to geodesic maths on an
    unknown reference -- the affine determinant already implies metres.
    """
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:25, 5:25] = True

    assert vectorize_mask(mask, TRANSFORM, min_area_m2=0.0).total_area_m2 == pytest.approx(
        40_000.0
    )
