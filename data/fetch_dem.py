"""
Fetches Copernicus DEM 30m elevation values matching each Sen1Floods11
chip's real geographic footprint, resampled onto the chip's exact pixel
grid, and saves the result alongside the other per-chip layers.

Copernicus DEM GLO-30 is distributed as one COG per 1-degree-by-1-degree
tile on AWS Open Data (public, no auth or API key needed), named by the
tile's southwest corner. Rather than downloading a full ~40MB tile for
every chip, this opens each remote tile directly and reads only the small
window overlapping a given chip -- GDAL's /vsicurl/ support (used
transparently by rasterio for http URLs) does this as a handful of HTTP
range requests instead of a full download. Chips sharing the same tile
(common -- 446 Sen1Floods11 chips need only ~50 distinct tiles) reuse one
open connection instead of reopening it per chip.
"""

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

DEM_BUCKET_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def dem_tile_url(lat: float, lon: float) -> str:
    """
    Copernicus DEM tiles are named by the integer degree cell containing
    (lat, lon) -- e.g. lat=9.43, lon=-1.15 falls in tile N09_00_W002_00
    (the tile spanning 9N-10N, 2W-1W; floor(-1.15) is -2, not -1).
    """
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    tile_name = f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"
    return f"{DEM_BUCKET_URL}/{tile_name}/{tile_name}.tif"


def fetch_dem_for_chip(
    chip_path: str | Path, tile_cache: dict[str, rasterio.DatasetReader]
) -> np.ndarray:
    """
    Read Copernicus DEM elevation values covering the same footprint and
    pixel grid as the given chip GeoTIFF, resampled to match it exactly so
    the result lines up pixel-for-pixel with the chip.

    `tile_cache` holds already-opened remote DEM tiles, keyed by URL, so
    repeated calls for chips sharing a tile don't reopen the connection
    each time. Pass the same dict across a batch of chips; the caller owns
    closing the datasets in it afterwards.
    """
    with rasterio.open(chip_path) as chip:
        chip_crs = chip.crs
        chip_transform = chip.transform
        chip_shape = (chip.height, chip.width)
        bounds = chip.bounds

    # Chips are a few km across, far smaller than a DEM tile's 1-degree
    # span, so sampling the tile at the chip's center is enough to find
    # the covering tile -- chips essentially never straddle a tile edge.
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2
    tile_url = dem_tile_url(center_lat, center_lon)

    if tile_url not in tile_cache:
        tile_cache[tile_url] = rasterio.open(tile_url)
    src = tile_cache[tile_url]

    dem = np.empty(chip_shape, dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dem,
        dst_transform=chip_transform,
        dst_crs=chip_crs,
        resampling=Resampling.bilinear,
    )
    return dem


def fetch_dem_for_all_chips(s1_dir: str | Path, dest_dir: str | Path) -> None:
    """
    Fetch and save a matching DEM chip for every *_S1Hand.tif file in
    `s1_dir`, writing `<chip_id>_DEMHand.tif` into `dest_dir`. Skips chips
    that already have a saved DEM file, so an interrupted run can resume.
    """
    s1_dir = Path(s1_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tile_cache: dict[str, rasterio.DatasetReader] = {}
    try:
        chip_paths = sorted(s1_dir.glob("*_S1Hand.tif"))
        for i, chip_path in enumerate(chip_paths, start=1):
            chip_id = chip_path.name.replace("_S1Hand.tif", "")
            dest_path = dest_dir / f"{chip_id}_DEMHand.tif"
            if dest_path.exists():
                continue

            dem = fetch_dem_for_chip(chip_path, tile_cache)

            with rasterio.open(chip_path) as chip:
                profile = chip.profile
            profile.update(count=1, dtype=np.float32)

            with rasterio.open(dest_path, "w", **profile) as dst:
                dst.write(dem, 1)

            print(f"[{i}/{len(chip_paths)}] {chip_id}")
    finally:
        for src in tile_cache.values():
            src.close()


if __name__ == "__main__":
    fetch_dem_for_all_chips(
        s1_dir="datasets/sen1floods11/S1Hand",
        dest_dir="datasets/sen1floods11/DEMHand",
    )
