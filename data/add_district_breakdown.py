"""
Adds a full 27-district breakdown to an already-built
frontend/public/data/assam_flood_demo.json, computed from the flood
polygons data/build_assam_statewide.py already produced -- no new
Sentinel-1/DEM/JRC fetches needed.

Why this is a separate pass rather than a field build_assam_statewide.py
writes directly: `inference.districts.summarize()` (shared with
inference/pipeline.py, the real scene-processing path) only reports the
top 5 *affected* districts by design, which is the right contract for a
grounded LLM tool result but not enough for the frontend report viewer to
look up "does district X specifically have real data" for an arbitrary
district. Recomputing `aggregate_to_districts` directly here (same
function, just not routed through the 5-item summary) gets every district,
including the honestly-zero and the genuinely-uncovered ones, without
changing what the shared summarize() contract returns anywhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference.districts import aggregate_to_districts, load_districts

DATA_DIR = Path("frontend/public/data")
JSON_PATH = DATA_DIR / "assam_flood_demo.json"
GEOJSON_PATH = DATA_DIR / "assam_flood_demo.geojson"


def main() -> None:
    summary = json.loads(JSON_PATH.read_text())
    flood_geojson = json.loads(GEOJSON_PATH.read_text())

    districts = load_districts()
    impacts = aggregate_to_districts(flood_geojson, districts=districts)

    covering = summary.get("tiles_covering_by_district", {})
    summary["districts"] = [
        {
            "name": impact.name,
            "flooded_hectares": round(impact.flooded_hectares, 1),
            "flooded_percent": round(impact.flooded_percent, 2),
            "tiles_covering": covering.get(impact.name, 0),
        }
        for impact in impacts
    ]

    JSON_PATH.write_text(json.dumps(summary, indent=2))
    covered = sum(1 for d in summary["districts"] if d["tiles_covering"] > 0)
    print(f"added full breakdown for {len(summary['districts'])} districts, {covered} with real coverage")


if __name__ == "__main__":
    main()
