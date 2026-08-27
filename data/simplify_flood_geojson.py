"""
Shrinks frontend/public/data/assam_flood_demo.geojson down to something a
browser can actually load.

The statewide merge (data/build_assam_statewide.py, 35 tiles' worth of
polygons at once) produced an 8,469-feature, ~100MB file, averaging ~260
vertices per polygon -- built for one small 16km AOI's
--simplify-pixel-fraction default (0.5), which turned out nowhere near
aggressive enough once merged across the whole state. Loading that file
live crashed the map: MapLibre runs all GeoJSON sources through a shared
worker, and this one source overloading it took the district/river/state
layers down too, not just the flood overlay -- confirmed live (nothing
rendered, not even the base map that worked before this file landed).

Re-simplifies with shapely (a much larger tolerance than a single-AOI
close-up view ever needed -- this is a state-zoom overlay, not a chip
inspector) and drops any polygon under a real-world area floor, since a
sub-hectare speck is invisible at state zoom and not worth the file size.
Does not touch the source raster, model, or aggregation logic -- it
operates on the already-computed geometry, so re-running this after any
future statewide build just needs the two path constants below.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.validation import make_valid

INPUT_PATH = Path("frontend/public/data/assam_flood_demo.geojson")
OUTPUT_PATH = INPUT_PATH

# In degrees -- roughly 300m at Assam's latitude. A first pass at 0.001deg
# (~100m) still left the file at 15.4MB (8,469 -> 4,170 features) --
# nowhere near light enough for a live public site to load eagerly, so
# this went further. State zoom shows the whole ~660x430km state in an
# ~900px-wide panel, where 300m is sub-pixel.
SIMPLIFY_TOLERANCE_DEG = 0.008
# Below this, a flooded patch is a few pixels at state zoom -- real, but
# not worth carrying its own polygon in a client-loaded file.
MIN_AREA_M2 = 500_000.0  # 50 hectares


def _geodesic_area_m2(geometry) -> float:
    from pyproj import Geod

    if geometry.is_empty:
        return 0.0
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geometry)
    return abs(float(area))


def main() -> None:
    data = json.loads(INPUT_PATH.read_text())
    before = len(data["features"])

    kept = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        if _geodesic_area_m2(geom) < MIN_AREA_M2:
            continue
        simplified = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
        if simplified.is_empty:
            continue
        # simplify(preserve_topology=True) still produced a few polygons
        # GEOS's own unary_union (aggregate_to_districts, run right after
        # this script) rejected with "unable to assign free hole to a
        # shell" -- a real invalid-geometry case, not a false alarm.
        # make_valid repairs it before it ever reaches that union.
        if not simplified.is_valid:
            simplified = make_valid(simplified)
        if simplified.geom_type not in ("Polygon", "MultiPolygon"):
            # make_valid can return a GeometryCollection mixing in stray
            # points/lines from a degenerate repair -- only the polygonal
            # part is a real flooded area; a fill layer can't render
            # anything else anyway.
            polys = [g for g in getattr(simplified, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                continue
            from shapely.ops import unary_union

            simplified = unary_union(polys)
        if simplified.is_empty:
            continue
        feature["geometry"] = mapping(simplified)
        kept.append(feature)

    data["features"] = kept

    # Full float64 repr (e.g. 90.11584519937598 -- 16 significant digits)
    # was most of the remaining file size, independent of vertex count:
    # simplification alone only took 8,469 features from 100MB to 11.3MB
    # before this. 5 decimal places is ~1.1m at this latitude, already far
    # finer than the 50m/px source data actually resolves.
    def round_coords(obj):
        # shapely's mapping() returns tuples, not lists -- checking only
        # for list here silently skipped every coordinate on the first
        # attempt (json.dumps happily serializes tuples as JSON arrays
        # too, so the bug produced no error, just no rounding at all).
        if isinstance(obj, (list, tuple)):
            if obj and isinstance(obj[0], (int, float)):
                return [round(v, 5) for v in obj]
            return [round_coords(v) for v in obj]
        return obj

    for feature in kept:
        feature["geometry"]["coordinates"] = round_coords(feature["geometry"]["coordinates"])

    OUTPUT_PATH.write_text(json.dumps(data, separators=(",", ":")))
    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"{before} -> {len(kept)} features, {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
