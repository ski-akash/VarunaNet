"""
Fetches real Assam river centerlines from OpenStreetMap (via the Overpass
API) and writes the simplified, filtered GeoJSON the frontend map renders
(`frontend/public/geo/assam_rivers.geojson`) -- the same "fetch once,
ship a pre-built static asset" pattern the district/state boundary files
already use (see MapView.tsx), rather than querying Overpass live from
the browser on every page load.

Filters down from every named `waterway=river` way in Assam (298 distinct
names in the real pull this was built against -- far too many to render
without becoming visual noise) to the Brahmaputra and its longest named
tributaries by real total length, not hand-picked. The Brahmaputra is
flagged `major: true` in the output so the frontend can render it in a
visually distinct color/width from its tributaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import requests
from shapely.geometry import mapping, shape
from shapely.ops import linemerge

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# A handful of real rivers appear under more than one spelling in OSM
# (diacritics, "River" suffix) -- merged here so they don't get counted,
# and ranked, as separate shorter rivers.
NAME_ALIASES = {
    "Barāk River": "Barak River",
    "Lohit River": "Lohit",
}


def fetch_raw_rivers(timeout: int = 90) -> list[dict]:
    query = f"""
    [out:json][timeout:{timeout}];
    area["name"="Assam"]["admin_level"="4"]->.assam;
    (
      way["waterway"="river"]["name"](area.assam);
    );
    out geom;
    """
    response = requests.post(OVERPASS_URL, data=query.encode(), timeout=timeout + 10)
    response.raise_for_status()
    return response.json()["elements"]


def build_rivers_geojson(elements: list[dict], top_n: int, simplify_tolerance_deg: float) -> dict:
    geoms_by_name: dict[str, list] = defaultdict(list)
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        name = el.get("tags", {}).get("name")
        if not name:
            continue
        name = NAME_ALIASES.get(name, name)
        coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
        if len(coords) < 2:
            continue
        geoms_by_name[name].append(shape({"type": "LineString", "coordinates": coords}))

    # Ranked by real total length (sum of every way segment sharing a
    # name), not hand-picked -- this is what makes "the 15 longest named
    # rivers" a real, reproducible filter rather than an editorial choice.
    ranked = sorted(geoms_by_name.items(), key=lambda item: -sum(g.length for g in item[1]))
    top_names = {name for name, _ in ranked[:top_n]}

    features = []
    for name in top_names:
        merged = linemerge(geoms_by_name[name])
        simplified = merged.simplify(simplify_tolerance_deg, preserve_topology=False)
        is_multi = simplified.geom_type == "MultiLineString"
        parts = list(simplified.geoms) if is_multi else [simplified]
        for part in parts:
            if part.length < 0.01:  # drops leftover slivers simplify() can produce
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {"name": name, "major": name == "Brahmaputra"},
                    "geometry": mapping(part),
                }
            )

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument(
        "--simplify-tolerance-deg",
        type=float,
        default=0.002,
        help="~220m at this latitude -- correct-looking at state zoom, far fewer vertices "
        "than the raw OSM geometry.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("frontend/public/geo/assam_rivers.geojson")
    )
    args = parser.parse_args()

    elements = fetch_raw_rivers()
    geojson = build_rivers_geojson(elements, args.top_n, args.simplify_tolerance_deg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(geojson))
    print(f"wrote {len(geojson['features'])} river features to {args.out}")


if __name__ == "__main__":
    main()
