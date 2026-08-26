"""
Tests for inference/districts.py.

Synthetic districts with known geometry, so expected areas and shares are
computable by hand rather than read off whatever the code produces. One
test runs against the real Assam boundaries when they are present.
"""

import json

import pytest

from inference.districts import (
    DEFAULT_DISTRICTS_PATH,
    DistrictImpact,
    aggregate_to_districts,
    load_districts,
    summarize,
    worst_affected,
)


def _square(x0, y0, side):
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [x0, y0],
                [x0 + side, y0],
                [x0 + side, y0 + side],
                [x0, y0 + side],
                [x0, y0],
            ]
        ],
    }


def _districts_file(tmp_path, entries):
    path = tmp_path / "districts.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"name": n}, "geometry": g}
                    for n, g in entries
                ],
            }
        )
    )
    return path


def _flood(*geometries):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": g} for g in geometries
        ],
    }


def test_a_flood_inside_one_district_is_attributed_to_it_alone(tmp_path):
    path = _districts_file(
        tmp_path, [("Alpha", _square(0, 0, 1.0)), ("Beta", _square(2, 0, 1.0))]
    )
    flood = _flood(_square(0.2, 0.2, 0.2))

    impacts = {i.name: i for i in aggregate_to_districts(flood, districts_path=path)}

    assert impacts["Alpha"].flooded_area_m2 > 0
    assert impacts["Beta"].flooded_area_m2 == 0.0


def test_a_flood_straddling_two_districts_is_split_between_them(tmp_path):
    """
    Floods follow river courses and routinely cross district lines. A
    centroid or bounding-box test would hand the whole flood to one
    district and report zero for its neighbour.
    """
    path = _districts_file(
        tmp_path, [("Left", _square(0, 0, 1.0)), ("Right", _square(1, 0, 1.0))]
    )
    # Straddles the shared edge at x=1, half in each.
    flood = _flood(_square(0.8, 0.2, 0.4))

    impacts = {i.name: i for i in aggregate_to_districts(flood, districts_path=path)}

    assert impacts["Left"].flooded_area_m2 > 0
    assert impacts["Right"].flooded_area_m2 > 0
    assert impacts["Left"].flooded_area_m2 == pytest.approx(
        impacts["Right"].flooded_area_m2, rel=0.02
    )


def test_overlapping_flood_polygons_are_not_double_counted(tmp_path):
    """
    Two overlapping polygons covering the same ground must not report
    twice that ground's area.
    """
    path = _districts_file(tmp_path, [("Alpha", _square(0, 0, 1.0))])

    single = aggregate_to_districts(_flood(_square(0.1, 0.1, 0.4)), districts_path=path)
    doubled = aggregate_to_districts(
        _flood(_square(0.1, 0.1, 0.4), _square(0.1, 0.1, 0.4)), districts_path=path
    )

    assert doubled[0].flooded_area_m2 == pytest.approx(single[0].flooded_area_m2)


def test_flooded_area_never_exceeds_the_district(tmp_path):
    path = _districts_file(tmp_path, [("Alpha", _square(0, 0, 1.0))])
    # A flood far larger than the district itself.
    flood = _flood(_square(-5, -5, 20.0))

    impact = aggregate_to_districts(flood, districts_path=path)[0]

    assert impact.flooded_area_m2 == pytest.approx(impact.district_area_m2, rel=1e-6)
    assert impact.flooded_fraction == pytest.approx(1.0, rel=1e-6)


def test_untouched_districts_are_reported_as_zero_not_omitted(tmp_path):
    """
    "Dhemaji is not flooded" is a real answer. A list that silently drops
    dry districts cannot distinguish them from districts the scene never
    covered.
    """
    path = _districts_file(
        tmp_path, [("Alpha", _square(0, 0, 1.0)), ("Beta", _square(50, 50, 1.0))]
    )
    impacts = aggregate_to_districts(_flood(_square(0.2, 0.2, 0.2)), districts_path=path)

    assert len(impacts) == 2
    assert {i.name for i in impacts} == {"Alpha", "Beta"}
    assert any(i.flooded_area_m2 == 0.0 for i in impacts)


def test_no_flood_at_all_reports_every_district_as_zero(tmp_path):
    path = _districts_file(tmp_path, [("Alpha", _square(0, 0, 1.0))])

    impacts = aggregate_to_districts({"type": "FeatureCollection", "features": []},
                                     districts_path=path)

    assert impacts[0].flooded_area_m2 == 0.0
    assert impacts[0].flooded_fraction == 0.0


def test_ranking_uses_share_of_district_not_absolute_area():
    """
    Assam's districts differ in size by more than 10x, so ranking by
    flooded km2 largely reproduces a ranking of district size. A small
    district half under water is the emergency.
    """
    big_but_barely = DistrictImpact("Big", district_area_m2=1000.0, flooded_area_m2=100.0)
    small_but_swamped = DistrictImpact("Small", district_area_m2=100.0, flooded_area_m2=60.0)

    ranked = worst_affected([big_but_barely, small_but_swamped])

    assert [i.name for i in ranked] == ["Small", "Big"]
    assert small_but_swamped.flooded_area_m2 < big_but_barely.flooded_area_m2


def test_summary_shape_is_json_serialisable_and_counts_correctly():
    impacts = [
        DistrictImpact("A", 1_000_000.0, 250_000.0),
        DistrictImpact("B", 1_000_000.0, 0.0),
    ]

    summary = summarize(impacts)

    assert summary["districts_total"] == 2
    assert summary["districts_affected"] == 1
    assert summary["worst_affected"][0]["name"] == "A"
    assert summary["worst_affected"][0]["flooded_percent"] == pytest.approx(25.0)
    json.dumps(summary)  # must not raise


def test_missing_district_file_raises_rather_than_reporting_no_flooding(tmp_path):
    with pytest.raises(FileNotFoundError, match="district boundaries"):
        load_districts(tmp_path / "nope.geojson")


@pytest.mark.skipif(
    not DEFAULT_DISTRICTS_PATH.exists(), reason="Assam boundaries not present"
)
def test_real_assam_boundaries_load_and_have_plausible_areas():
    districts = load_districts()

    assert len(districts) == 27, "Assam has 27 districts in the 2011 Census extract"

    impacts = aggregate_to_districts(
        {"type": "FeatureCollection", "features": []}, districts=districts
    )
    total_km2 = sum(i.district_area_m2 for i in impacts) / 1e6
    # Assam is ~78,438 km2. Allow generous slack for boundary simplification.
    assert 70_000 < total_km2 < 90_000, f"total area {total_km2:.0f} km2 is implausible"
