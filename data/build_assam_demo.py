"""
The rest of Track B Step 3/gate (spec section 15.4): turns the raw
Sentinel-1 pair `data/fetch_assam_scene.py` pulls into a real district-level
flood result, using this project's own already-validated classical
baseline (per-scene Otsu + HAND/slope masking + permanent-water removal --
spec section 15.2's Step 1 pipeline), not the untrained local demo CNN
checkpoint. The demo CNN (inference/stage_demo_scene.py) has random
weights, so it would report zero flood everywhere on real data; Otsu+HAND
has real, measured accuracy characteristics on Sen1Floods11
(benchmarks/RESULTS.md), which makes it the honest choice for a first
real-Assam result, not a downgrade.

Auxiliary layers (DEM for slope/HAND, JRC Global Surface Water for
permanent-water removal) are pulled from the same Earth Engine session as
the Sentinel-1 pair, over the identical AOI/transform, so every layer is
already co-registered pixel-for-pixel -- no separate reprojection step.

Output: a district-level summary JSON at
frontend/public/data/assam_flood_demo.json, and the flood polygons as
GeoJSON at frontend/public/data/assam_flood_demo.geojson, both served
statically by the existing Vite dev/build setup the same way
assam_districts.geojson already is.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from benchmarks.otsu import compute_otsu_threshold, smooth_backscatter
from benchmarks.otsu_hand import otsu_hand_water_mask
from data.fetch_assam_scene import download_geotiff, init_earth_engine
from data.hand import compute_hand
from data.permanent_water import compute_flood_extent, compute_permanent_water_mask
from data.terrain import compute_slope
from inference.districts import aggregate_to_districts, summarize
from inference.vectorize import to_feature_collection, vectorize_mask

# Same tuned thresholds spec section 15.2 Step 1 already found by grid
# search on the Sen1Floods11 train split (benchmarks/tune_otsu_hand.py) --
# reused here rather than re-tuned, since there's no Assam-specific labeled
# data yet to tune against, and these are this project's best real evidence
# for what HAND/slope cutoffs distinguish real flood water from look-alikes.
HAND_THRESHOLD_M = 5.0
SLOPE_THRESHOLD_DEG = 15.0
# 10m matches the scale requested from Earth Engine (data/fetch_assam_scene.py's
# default) -- same not-a-tuned-constant approximation
# data/chip_terrain.CHIP_PIXEL_SIZE_M already uses for Sen1Floods11 chips,
# which are also stored in degrees (EPSG:4326) with no exact meters/pixel.
PIXEL_SIZE_M = 10.0
HAND_ACCUMULATION_THRESHOLD = 100
JRC_COLLECTION = "JRC/GSW1_4/GlobalSurfaceWater"


def read_vv_vh(path: Path) -> tuple[np.ndarray, "rasterio.Affine", object]:
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs


def fetch_dem(
    ee_project: str, aoi_bounds: tuple[float, float, float, float], scale: float
) -> np.ndarray:
    import ee

    init_earth_engine(ee_project)
    aoi = ee.Geometry.Rectangle(list(aoi_bounds))
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(aoi).select("DEM").mosaic()
    arr, _, _ = download_geotiff(dem, aoi, ["DEM"], scale)
    return arr[0]


def fetch_jrc_occurrence(
    ee_project: str, aoi_bounds: tuple[float, float, float, float], scale: float
) -> np.ndarray:
    """
    JRC Global Surface Water's `occurrence` band: percent of Landsat
    observations 1984-2021 that found water at each pixel -- the real,
    remote-sensed permanent-water reference spec section 3.2 calls for,
    not the already-binary Sen1Floods11-shipped JRCWaterHand chips this
    project's other Otsu+permanent-water baseline uses (those are a
    different, pre-thresholded product; this is the raw occurrence layer
    data/permanent_water.py's compute_permanent_water_mask was actually
    built to threshold).
    """
    import ee

    init_earth_engine(ee_project)
    aoi = ee.Geometry.Rectangle(list(aoi_bounds))
    occurrence = ee.Image(JRC_COLLECTION).select("occurrence").unmask(0)
    arr, _, _ = download_geotiff(occurrence, aoi, ["occurrence"], scale)
    return arr[0].astype(np.float64)


def build_demo(
    ee_project: str,
    during_path: Path,
    dry_path: Path,
    aoi_bounds: tuple[float, float, float, float],
    scale: float,
    min_area_m2: float,
    simplify_pixel_fraction: float = 0.5,
) -> dict:
    during, transform, crs = read_vv_vh(during_path)
    vh = during[1]  # VV_db, VH_db -- same _OTSU_BAND_INDEX=1 convention as benchmarks/evaluate.py

    print("fetching DEM...")
    dem = fetch_dem(ee_project, aoi_bounds, scale)
    print("computing slope...")
    slope = compute_slope(dem, PIXEL_SIZE_M)
    print("computing HAND (flow routing over a real scene -- can take a while)...")
    hand = compute_hand(dem, PIXEL_SIZE_M, accumulation_threshold=HAND_ACCUMULATION_THRESHOLD)

    print("fetching JRC Global Surface Water occurrence...")
    occurrence = fetch_jrc_occurrence(ee_project, aoi_bounds, scale)
    permanent_water = compute_permanent_water_mask(occurrence)

    smoothed_vh = smooth_backscatter(vh)
    finite = smoothed_vh[np.isfinite(smoothed_vh)]
    threshold = compute_otsu_threshold(finite)
    print(f"Otsu threshold (VH, dB): {threshold:.2f}")

    water = otsu_hand_water_mask(
        smoothed_vh,
        hand,
        slope,
        otsu_threshold=threshold,
        hand_threshold=HAND_THRESHOLD_M,
        slope_threshold=SLOPE_THRESHOLD_DEG,
    )
    flood_extent = compute_flood_extent(water, permanent_water)

    print(f"water fraction (pre permanent-water removal): {water.mean():.4f}")
    print(f"flood fraction (post permanent-water removal): {flood_extent.mean():.4f}")

    vectorized = vectorize_mask(
        flood_extent,
        transform,
        crs=crs,
        min_area_m2=min_area_m2,
        simplify_pixel_fraction=simplify_pixel_fraction,
    )
    flood_geojson = to_feature_collection(vectorized)
    impacts = aggregate_to_districts(flood_geojson)
    district_summary = summarize(impacts)

    return {
        "scene_id": during_path.stem.replace("_vv_vh", ""),
        "dry_reference_scene_id": dry_path.stem.replace("_vv_vh", ""),
        "source": (
            "COPERNICUS/S1_GRD (Google Earth Engine), sigma0 -- see data/fetch_assam_scene.py"
        ),
        "method": (
            f"Otsu (VH, threshold={threshold:.2f} dB) + HAND<={HAND_THRESHOLD_M}m/"
            f"slope<={SLOPE_THRESHOLD_DEG}deg masking + JRC permanent-water removal "
            "(classical baseline, not the CNN -- see this file's module docstring)"
        ),
        "aoi_bounds": list(aoi_bounds),
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "water_pixel_fraction": round(float(water.mean()), 6),
        "flood_pixel_fraction": round(float(flood_extent.mean()), 6),
        "flood_polygons": len(vectorized.polygons),
        "specks_dropped": vectorized.dropped_as_speck,
        **district_summary,
        "flood_geojson": flood_geojson,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-project", default=os.environ.get("EE_PROJECT_ID"))
    parser.add_argument("--during", type=Path, required=True)
    parser.add_argument("--dry", type=Path, required=True)
    parser.add_argument("--west", type=float, default=94.15)
    parser.add_argument("--south", type=float, default=27.0)
    parser.add_argument("--east", type=float, default=94.30)
    parser.add_argument("--north", type=float, default=27.15)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--min-area-m2", type=float, default=10_000.0)
    parser.add_argument(
        "--simplify-pixel-fraction",
        type=float,
        default=0.5,
        help="Higher = coarser polygons, smaller file. 0.5 (project default) keeps sub-pixel "
        "accuracy; a served demo overlay viewed at state/district zoom doesn't need that, so "
        "a larger value is a reasonable choice specifically for this file's real output size.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("frontend/public/data"))
    args = parser.parse_args()

    if not args.ee_project:
        raise SystemExit("Set --ee-project or EE_PROJECT_ID")

    result = build_demo(
        args.ee_project,
        args.during,
        args.dry,
        (args.west, args.south, args.east, args.north),
        args.scale,
        args.min_area_m2,
        args.simplify_pixel_fraction,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flood_geojson = result.pop("flood_geojson")
    (args.out_dir / "assam_flood_demo.json").write_text(json.dumps(result, indent=2))
    (args.out_dir / "assam_flood_demo.geojson").write_text(json.dumps(flood_geojson, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
