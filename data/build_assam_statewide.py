"""
Extends the single-AOI proof in data/build_assam_demo.py to the whole of
Assam, not just the ~16km tile near the Lakhimpur/Jorhat border.

**Why this is a separate script, not a bigger --west/--south/--east/--north
on the existing one**: data/fetch_assam_scene.py's download is Earth
Engine's *synchronous* getDownloadURL, capped at 50MB per request -- hit
directly on this project's first real AOI attempt (a 0.6x0.7deg box
requested 937MB and failed). One request cannot cover the state. This
script instead tiles the state into a grid of AOIs, each small enough to
stay under that cap, runs the same per-tile pipeline
(fetch pair -> Otsu+HAND+permanent-water -> vectorize) build_assam_demo.py
already validated on one tile, and merges every tile's flood polygons into
one district aggregation at the end -- aggregate_to_districts takes a
single flood_geojson and unions overlapping geometry itself, so collecting
polygons from N tiles and aggregating once is equivalent to (and cheaper
than) aggregating per tile and summing.

**Resolution traded down from 10m to 50m to make the tile count tractable**:
at 10m, a tile under the 50MB cap is only ~16km across, and Assam is
~660x430km -- over a thousand tiles, each needing its own orbit-matched
Sentinel-1 pair. At 50m, a tile can be ~115km across at the same cap,
bringing the state down to ~24 tiles. Flood polygons come out coarser at
the district-boundary edges as a result -- an explicit, named tradeoff, not
a silent quality loss.

Districts with no Sentinel-1 coverage in the requested date windows (no
orbit-matched pair found), or entirely outside the tile grid's footprint,
still report zero flooded area but are distinguished by `tiles_covering`
in the output -- 0 means genuinely no data, not "checked and dry", same
honesty rule as the single-AOI script's district output already follows.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.otsu import compute_otsu_threshold, smooth_backscatter
from benchmarks.otsu_hand import otsu_hand_water_mask
from data.build_assam_demo import (
    HAND_ACCUMULATION_THRESHOLD,
    HAND_THRESHOLD_M,
    SLOPE_THRESHOLD_DEG,
    fetch_dem,
    fetch_jrc_occurrence,
)
from data.fetch_assam_scene import fetch_scene_stack
from data.hand import compute_hand
from data.permanent_water import compute_flood_extent, compute_permanent_water_mask
from data.terrain import compute_slope
from inference.districts import aggregate_to_districts, load_districts, summarize
from inference.vectorize import to_feature_collection, vectorize_mask

DISTRICTS_PATH = Path("frontend/public/geo/assam_districts.geojson")

# A first 1.05deg tile at 50m/px came back as a real 98.5MB request --
# nearly double what the raw pixel-byte math predicted (GEE's actual
# synchronous-download overhead is higher than a plain width*height*bands*
# bytes estimate), and over the real ~48MB (50331648 byte) cap. 0.65deg
# was then tried directly and confirmed live: a real download at this size
# succeeds (1449x1448px, ~16.8MB of raw pixel bytes, safely under cap even
# accounting for the same overhead ratio).
TILE_SIZE_DEG = 0.65
SCALE_M = 50.0


def make_tile_grid(bounds: tuple[float, float, float, float], tile_size_deg: float) -> list[tuple[float, float, float, float]]:
    """
    Covers `bounds` with a regular grid of (west, south, east, north)
    tiles of side `tile_size_deg`, clipped to `bounds` at the state's own
    edges rather than overshooting into neighbouring states -- Sentinel-1
    coverage there wouldn't be wrong, just wasted requests for districts
    that were never going to be inside them.
    """
    west, south, east, north = bounds
    tiles = []
    lon = west
    while lon < east:
        lat = south
        tile_east = min(lon + tile_size_deg, east)
        while lat < north:
            tile_north = min(lat + tile_size_deg, north)
            tiles.append((lon, lat, tile_east, tile_north))
            lat += tile_size_deg
        lon += tile_size_deg
    return tiles


def assam_bounds_from_districts(districts_path: Path = DISTRICTS_PATH) -> tuple[float, float, float, float]:
    collection = json.loads(districts_path.read_text())
    xs: list[float] = []
    ys: list[float] = []

    def walk(coords):
        if isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for c in coords:
                walk(c)

    for feature in collection["features"]:
        walk(feature["geometry"]["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def process_tile(
    ee_project: str,
    aoi_bounds: tuple[float, float, float, float],
    during_start: str,
    during_end: str,
    dry_start: str,
    dry_end: str,
    min_area_m2: float,
    simplify_pixel_fraction: float,
) -> dict | None:
    """
    Runs the same per-tile pipeline build_assam_demo.build_demo already
    validated, but returns raw flood-polygon features instead of an
    aggregated district summary -- aggregation happens once, across every
    tile's polygons together, in main().

    Returns None (not an error) when this tile genuinely has no
    Sentinel-1 pair in the requested windows -- a real, expected outcome
    for tiles over neighbouring states or with a data gap, not a bug.
    """
    try:
        stack = fetch_scene_stack(
            ee_project, aoi_bounds, during_start, during_end, dry_start, dry_end, scale=SCALE_M
        )
    except ValueError as exc:
        print(f"  skipping tile {aoi_bounds}: {exc}")
        return None

    during_arr = stack["during_vv_vh"]
    transform = stack["transform"]
    crs = stack["crs"]
    vh = during_arr[1]

    dem = fetch_dem(ee_project, aoi_bounds, SCALE_M)
    slope = compute_slope(dem, SCALE_M)
    hand = compute_hand(dem, SCALE_M, accumulation_threshold=HAND_ACCUMULATION_THRESHOLD)

    occurrence = fetch_jrc_occurrence(ee_project, aoi_bounds, SCALE_M)
    permanent_water = compute_permanent_water_mask(occurrence)

    smoothed_vh = smooth_backscatter(vh)
    finite = smoothed_vh[np.isfinite(smoothed_vh)]
    if finite.size == 0:
        print(f"  skipping tile {aoi_bounds}: no finite VH pixels (no real coverage)")
        return None
    threshold = compute_otsu_threshold(finite)

    water = otsu_hand_water_mask(
        smoothed_vh,
        hand,
        slope,
        otsu_threshold=threshold,
        hand_threshold=HAND_THRESHOLD_M,
        slope_threshold=SLOPE_THRESHOLD_DEG,
    )
    flood_extent = compute_flood_extent(water, permanent_water)

    vectorized = vectorize_mask(
        flood_extent,
        transform,
        crs=crs,
        min_area_m2=min_area_m2,
        simplify_pixel_fraction=simplify_pixel_fraction,
    )
    flood_geojson = to_feature_collection(vectorized)

    return {
        "aoi_bounds": list(aoi_bounds),
        "during_scene_id": stack["during_scene_id"],
        "dry_scene_id": stack["dry_scene_id"],
        "otsu_threshold_db": round(float(threshold), 2),
        "flood_pixel_fraction": round(float(flood_extent.mean()), 6),
        "flood_polygons": len(vectorized.polygons),
        "flood_geojson": flood_geojson,
    }


def build_statewide(
    ee_project: str,
    during_start: str,
    during_end: str,
    dry_start: str,
    dry_end: str,
    min_area_m2: float,
    simplify_pixel_fraction: float,
    tile_size_deg: float,
) -> dict:
    bounds = assam_bounds_from_districts()
    all_tiles = make_tile_grid(bounds, tile_size_deg)

    # Drop tiles that don't actually touch any district -- the grid is a
    # regular rectangle over the state's bounding box, but Assam's own
    # shape is not a rectangle, so corner/edge tiles routinely cover only
    # neighbouring states or open grid slack. Fetching those would spend
    # real Sentinel-1/DEM/JRC quota on a tile no district could ever use.
    from shapely.geometry import box as shapely_box

    districts = load_districts(DISTRICTS_PATH)
    tiles = [
        tile
        for tile in all_tiles
        if any(geometry.intersects(shapely_box(*tile)) for _, geometry in districts)
    ]
    print(f"state bounds: {bounds}")
    print(
        f"{len(tiles)}/{len(all_tiles)} tiles at {tile_size_deg}deg "
        f"(~{tile_size_deg * 111:.0f}km) each actually touch a district"
    )

    all_features: list[dict] = []
    tile_results: list[dict] = []
    for i, tile in enumerate(tiles, start=1):
        print(f"[{i}/{len(tiles)}] tile {tile}")
        try:
            result = process_tile(
                ee_project, tile, during_start, during_end, dry_start, dry_end,
                min_area_m2, simplify_pixel_fraction,
            )
        except Exception as exc:  # noqa: BLE001 -- a real 35-tile GEE run
            # already died once on a transient network error (a 503 that
            # download_geotiff now retries, but this is the backstop for
            # whatever the next flaky failure turns out to be). One bad
            # tile skipping rather than discarding every tile already
            # completed before it is the right tradeoff for a run this
            # long -- the honest "0 tiles covering" signal in the output
            # still shows which districts this tile would have covered.
            print(f"  tile {tile} failed, skipping: {exc!r}")
            continue
        if result is None:
            continue
        all_features.extend(result["flood_geojson"]["features"])
        tile_results.append(
            {k: v for k, v in result.items() if k != "flood_geojson"}
        )

    merged_geojson = {"type": "FeatureCollection", "features": all_features}
    impacts = aggregate_to_districts(merged_geojson, districts=districts)
    district_summary = summarize(impacts)

    covering_counts: dict[str, int] = {name: 0 for name, _ in districts}

    for tile_result in tile_results:
        w, s, e, n = tile_result["aoi_bounds"]
        tile_box = shapely_box(w, s, e, n)
        for name, geometry in districts:
            if geometry.intersects(tile_box):
                covering_counts[name] += 1

    return {
        "source": (
            "COPERNICUS/S1_GRD (Google Earth Engine), sigma0 -- tiled statewide fetch, "
            "see data/build_assam_statewide.py"
        ),
        "method": (
            f"Otsu (per-tile VH threshold) + HAND<={HAND_THRESHOLD_M}m/"
            f"slope<={SLOPE_THRESHOLD_DEG}deg masking + JRC permanent-water removal, "
            f"{len(tile_results)}/{len(tiles)} tiles at {SCALE_M:.0f}m/px covered "
            "(classical baseline, not the CNN)"
        ),
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiles_total": len(tiles),
        "tiles_covered": len(tile_results),
        "tile_results": tile_results,
        "tiles_covering_by_district": covering_counts,
        "flood_polygons": len(all_features),
        **district_summary,
        "flood_geojson": merged_geojson,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-project", default=os.environ.get("EE_PROJECT_ID"))
    parser.add_argument("--during-start", default="2020-07-10")
    parser.add_argument("--during-end", default="2020-07-25")
    parser.add_argument("--dry-start", default="2020-01-01")
    parser.add_argument("--dry-end", default="2020-04-30")
    parser.add_argument("--min-area-m2", type=float, default=40_000.0)
    parser.add_argument("--simplify-pixel-fraction", type=float, default=0.5)
    parser.add_argument("--tile-size-deg", type=float, default=TILE_SIZE_DEG)
    parser.add_argument("--out-dir", type=Path, default=Path("frontend/public/data"))
    args = parser.parse_args()

    if not args.ee_project:
        raise SystemExit("Set --ee-project or EE_PROJECT_ID")

    result = build_statewide(
        args.ee_project,
        args.during_start,
        args.during_end,
        args.dry_start,
        args.dry_end,
        args.min_area_m2,
        args.simplify_pixel_fraction,
        args.tile_size_deg,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flood_geojson = result.pop("flood_geojson")
    (args.out_dir / "assam_flood_demo.json").write_text(json.dumps(result, indent=2))
    (args.out_dir / "assam_flood_demo.geojson").write_text(json.dumps(flood_geojson, indent=2))

    print(json.dumps({k: v for k, v in result.items() if k != "tile_results"}, indent=2))


if __name__ == "__main__":
    main()
