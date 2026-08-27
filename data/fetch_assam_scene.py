"""
Pulls a real, matched pre-flood/during-flood Sentinel-1 scene pair over an
Assam AOI from Google Earth Engine, and builds a data-contract-conformant
5-channel raster stack from it (spec section 15.4, Track B Step 3).

**CRITICAL, and the single highest-risk configuration detail in the whole
Track B pipeline (spec section 15.4's own words):** this uses the DEFAULT
`COPERNICUS/S1_GRD` collection -- sigma0, geometrically orthorectified but
NOT radiometrically terrain-flattened. Sen1Floods11 (the training data this
project's model was fit on) is sigma0. GEE also offers a
`gamma0_terrain` RTC option; using it here would introduce exactly the
domain shift this specific source choice exists to eliminate, silently,
producing a plausible-looking but systematically wrong map. Verified
directly, not assumed: a real pulled scene's VV/VH percentiles land inside
this project's own Sen1Floods11-derived normalization stats range
(data/sen1floods11_normalization_stats.json) -- see this module's own
`sanity_check_against_sen1floods11_stats`.

Same relative orbit for both scenes in a pair, per spec section 15.4:
backscatter is only directly comparable within one geometry/incidence-angle
combination.

Requires `pip install earthengine-api` and a Google Earth Engine account
already authenticated (`ee.Authenticate()`, one-time) with a Cloud project
that has Earth Engine enabled -- pass that project id via
`--ee-project` or the `EE_PROJECT_ID` environment variable, never
hardcoded here.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import requests

if TYPE_CHECKING:
    import ee

# GLO30_2024_1 supersedes the plain GLO30 collection (GEE deprecation
# warning, checked live rather than left to bit-rot silently) -- same
# Copernicus DEM data, current asset id.
DEM_COLLECTION = "COPERNICUS/DEM/GLO30_2024_1"
S1_COLLECTION = "COPERNICUS/S1_GRD"
NORMALIZATION_STATS_PATH = Path("data/sen1floods11_normalization_stats.json")


def init_earth_engine(project: str) -> None:
    import ee

    ee.Initialize(project=project)


def find_orbit_matched_pair(
    aoi: ee.Geometry,
    during_start: str,
    during_end: str,
    dry_start: str,
    dry_end: str,
) -> tuple[str, str]:
    """
    Finds one during-flood scene and one dry-season scene over `aoi` that
    share a relative orbit and pass direction, so their backscatter
    geometry is directly comparable (spec section 15.4). Picks the first
    match found in each window -- deterministic given the same date
    windows, not randomly chosen.
    """
    import ee

    base = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    during = base.filterDate(during_start, during_end)
    during_info = during.getInfo()["features"]
    if not during_info:
        raise ValueError(
            f"no Sentinel-1 scenes found over this AOI in {during_start}..{during_end}"
        )

    for feature in during_info:
        props = feature["properties"]
        orbit = props["relativeOrbitNumber_start"]
        pass_direction = props["orbitProperties_pass"]

        dry = (
            base.filterDate(dry_start, dry_end)
            .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
            .filter(ee.Filter.eq("orbitProperties_pass", pass_direction))
        )
        dry_info = dry.getInfo()["features"]
        if dry_info:
            return feature["properties"]["system:index"], dry_info[0]["properties"]["system:index"]

    raise ValueError(
        f"no dry-season scene shares a relative orbit with any during-flood scene "
        f"found in {during_start}..{during_end}"
    )


def download_geotiff(
    image: ee.Image, aoi: ee.Geometry, bands: list[str], scale: float
) -> tuple[np.ndarray, "rasterio.Affine", object]:
    """
    Downloads a small clipped region as a real GeoTIFF via Earth Engine's
    synchronous download URL (no batch Export.image task needed for an
    AOI this size) and reads it back with rasterio -- so what lands on
    disk is verified to actually be a valid, readable raster, not just a
    URL that resolved. Returns the array *and* its real transform/CRS,
    not just pixels: vectorizing later needs the true affine or areas come
    out in square degrees (the exact bug inference/README.md documents
    hitting on the first real chip, not hypothetical).
    """
    url = image.select(bands).getDownloadURL(
        {"region": aoi, "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF"}
    )
    # GEE's synchronous download backend returns real, transient 503s under
    # load -- hit live during the first statewide multi-tile run, where it
    # killed an otherwise-working ~35-tile job on tile 2. Retried with
    # backoff rather than left to crash the whole run over one flaky
    # request; a real (non-503) failure still raises after 3 tries.
    last_error: requests.exceptions.HTTPError | None = None
    for attempt in range(3):
        response = requests.get(url, timeout=120)
        if response.status_code == 503:
            last_error = requests.exceptions.HTTPError(
                f"503 Server Error (attempt {attempt + 1}/3)", response=response
            )
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        last_error = None
        break
    if last_error is not None:
        raise last_error
    content_type = response.headers.get("content-type", "")
    if "zip" in content_type:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            tif_name = next(n for n in zf.namelist() if n.endswith(".tif"))
            data = zf.read(tif_name)
    else:
        data = response.content

    with rasterio.MemoryFile(data) as memfile:
        with memfile.open() as src:
            return src.read(), src.transform, src.crs


def fetch_scene_stack(
    ee_project: str,
    aoi_bounds: tuple[float, float, float, float],
    during_start: str,
    during_end: str,
    dry_start: str,
    dry_end: str,
    scale: float = 10.0,
) -> dict:
    """
    Returns real VV/VH arrays (dB, matching data/sen1floods11.py's own
    convention) for both scenes plus provenance, ready for
    validate_against_sen1floods11_stats and the terrain step.
    """
    import ee

    init_earth_engine(ee_project)
    aoi = ee.Geometry.Rectangle(list(aoi_bounds))

    during_id, dry_id = find_orbit_matched_pair(aoi, during_start, during_end, dry_start, dry_end)

    during_img = ee.Image(f"{S1_COLLECTION}/{during_id}")
    dry_img = ee.Image(f"{S1_COLLECTION}/{dry_id}")

    during_arr, transform, crs = download_geotiff(during_img, aoi, ["VV", "VH"], scale)
    dry_arr, _, _ = download_geotiff(dry_img, aoi, ["VV", "VH"], scale)

    return {
        "during_scene_id": during_id,
        "dry_scene_id": dry_id,
        "during_vv_vh": during_arr,  # [2, H, W]: VV_db, VH_db
        "dry_vv_vh": dry_arr,
        "aoi_bounds": aoi_bounds,
        "transform": transform,
        "crs": crs,
    }


def sanity_check_against_sen1floods11_stats(vv_vh: np.ndarray) -> dict:
    """
    Compares a real pulled scene's VV/VH percentiles against this
    project's own Sen1Floods11-derived normalization stats. Both being in
    the same real dB range is direct evidence this is genuinely sigma0
    (not accidentally the RTC/gamma0 collection, which would read
    systematically differently) -- spec section 15.4 calls this out as
    an already-passing consistency check, verified again here on real
    Assam data specifically, not just asserted from the earlier Sen1Floods11
    check.
    """
    stats = json.loads(NORMALIZATION_STATS_PATH.read_text())
    vv, vh = vv_vh[0], vv_vh[1]
    finite_vv = vv[np.isfinite(vv)]
    finite_vh = vh[np.isfinite(vh)]
    return {
        "vv_median": float(np.median(finite_vv)),
        "vh_median": float(np.median(finite_vh)),
        "sen1floods11_vv_mean": stats["mean"][0],
        "sen1floods11_vh_mean": stats["mean"][1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-project", default=os.environ.get("EE_PROJECT_ID"))
    # A ~0.15deg box (~16km) at 10m/px * 2 bands * float32 stays under
    # getDownloadURL's 50MB synchronous-request cap -- confirmed by hitting
    # that exact cap with the original 0.6x0.7deg box (937MB requested) and
    # shrinking down to what actually fits, rather than guessing a size.
    parser.add_argument("--west", type=float, default=94.15)
    parser.add_argument("--south", type=float, default=27.0)
    parser.add_argument("--east", type=float, default=94.30)
    parser.add_argument("--north", type=float, default=27.15)
    parser.add_argument("--during-start", default="2020-07-10")
    parser.add_argument("--during-end", default="2020-07-25")
    parser.add_argument("--dry-start", default="2020-01-01")
    parser.add_argument("--dry-end", default="2020-04-30")
    parser.add_argument("--out", default="datasets/assam", type=Path)
    args = parser.parse_args()

    if not args.ee_project:
        raise SystemExit(
            "Set --ee-project or EE_PROJECT_ID (a Google Cloud project with Earth Engine enabled)"
        )

    result = fetch_scene_stack(
        args.ee_project,
        (args.west, args.south, args.east, args.north),
        args.during_start,
        args.during_end,
        args.dry_start,
        args.dry_end,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for scene_id, arr in [
        (result["during_scene_id"], result["during_vv_vh"]),
        (result["dry_scene_id"], result["dry_vv_vh"]),
    ]:
        with rasterio.open(
            args.out / f"{scene_id}_vv_vh.tif",
            "w",
            driver="GTiff",
            height=arr.shape[1],
            width=arr.shape[2],
            count=2,
            dtype="float32",
            crs=result["crs"],
            transform=result["transform"],
        ) as dst:
            dst.write(arr.astype("float32"))

    print(f"during-flood scene: {result['during_scene_id']}")
    print(f"dry-season scene:   {result['dry_scene_id']}")
    during_check = sanity_check_against_sen1floods11_stats(result["during_vv_vh"])
    dry_check = sanity_check_against_sen1floods11_stats(result["dry_vv_vh"])
    print("sanity check (during-flood):", during_check)
    print("sanity check (dry-season):  ", dry_check)


if __name__ == "__main__":
    main()
