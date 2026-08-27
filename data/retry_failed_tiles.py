"""
One-off: retries the 8 tiles that failed in the first statewide run with
repeated SSLError(CERTIFICATE_VERIFY_FAILED for oauth2.googleapis.com) --
a network interruption partway through that run (the pattern of 8
consecutive identical cert-hostname-mismatch errors right after 27 clean
tiles points to the machine's network changing mid-run, not a real GEE or
code problem), not something data/fetch_assam_scene.py's existing 503
retry was built to catch. Merges their flood polygons into the existing
frontend/public/data/assam_flood_demo.json/.geojson rather than
re-running all 35 tiles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from data.build_assam_statewide import process_tile
from inference.districts import aggregate_to_districts, load_districts, summarize

FAILED_TILES = [
    (93.60506000000004, 27.384299999999993, 94.25506000000004, 27.97203),
    (94.25506000000004, 26.084299999999995, 94.90506000000005, 26.734299999999994),
    (94.25506000000004, 26.734299999999994, 94.90506000000005, 27.384299999999993),
    (94.25506000000004, 27.384299999999993, 94.90506000000005, 27.97203),
    (94.90506000000005, 26.734299999999994, 95.55506000000005, 27.384299999999993),
    (94.90506000000005, 27.384299999999993, 95.55506000000005, 27.97203),
    (95.55506000000005, 26.734299999999994, 96.02104, 27.384299999999993),
    (95.55506000000005, 27.384299999999993, 96.02104, 27.97203),
]

DATA_DIR = Path("frontend/public/data")
JSON_PATH = DATA_DIR / "assam_flood_demo.json"
GEOJSON_PATH = DATA_DIR / "assam_flood_demo.geojson"


def main() -> None:
    ee_project = os.environ.get("EE_PROJECT_ID")
    if not ee_project:
        raise SystemExit("Set EE_PROJECT_ID")

    summary = json.loads(JSON_PATH.read_text())
    flood_geojson = json.loads(GEOJSON_PATH.read_text())

    new_tile_results = []
    for i, tile in enumerate(FAILED_TILES, start=1):
        print(f"[{i}/{len(FAILED_TILES)}] retrying tile {tile}")
        try:
            result = process_tile(
                ee_project, tile,
                "2020-07-10", "2020-07-25", "2020-01-01", "2020-04-30",
                40_000.0, 0.5,
            )
        except Exception as exc:  # noqa: BLE001 -- same backstop as the main loop
            print(f"  tile {tile} failed again, skipping: {exc!r}")
            continue
        if result is None:
            continue
        flood_geojson["features"].extend(result["flood_geojson"]["features"])
        new_tile_results.append({k: v for k, v in result.items() if k != "flood_geojson"})

    if not new_tile_results:
        print("no additional tiles succeeded -- nothing to merge")
        return

    districts = load_districts()
    impacts = aggregate_to_districts(flood_geojson, districts=districts)
    district_summary = summarize(impacts)

    from shapely.geometry import box as shapely_box

    covering = {name: 0 for name, _ in districts}
    all_tile_bounds = [tuple(t["aoi_bounds"]) for t in summary["tile_results"]] + [
        tuple(t["aoi_bounds"]) for t in new_tile_results
    ]
    for bounds in all_tile_bounds:
        tile_box = shapely_box(*bounds)
        for name, geometry in districts:
            if geometry.intersects(tile_box):
                covering[name] += 1

    summary["tile_results"].extend(new_tile_results)
    summary["tiles_covered"] = len(summary["tile_results"])
    summary["tiles_covering_by_district"] = covering
    summary["flood_polygons"] = len(flood_geojson["features"])
    summary["method"] = summary["method"].split(",")[0] + (
        f", {summary['tiles_covered']}/{summary['tiles_total']} tiles at 50m/px covered "
        "(classical baseline, not the CNN)"
    )
    summary.update(district_summary)

    JSON_PATH.write_text(json.dumps(summary, indent=2))
    GEOJSON_PATH.write_text(json.dumps(flood_geojson, indent=2))
    print(f"merged {len(new_tile_results)} more tiles; now {summary['tiles_covered']}/{summary['tiles_total']}")


if __name__ == "__main__":
    main()
