"""
Stages one real Sen1Floods11 chip as a scene inference/service.py can serve,
so the full stack (frontend -> gateway -> Redis -> Postgres -> this service)
can be exercised locally against real data before real Assam Sentinel-1
ingestion exists (spec section 15.4, blocked on Earth Engine credentials
this environment doesn't have).

Deliberately reuses the exact same loading path training uses
(Sen1Floods11Dataset, data/chip_terrain.get_terrain,
data/normalization.apply_normalization) rather than re-deriving it, for the
same reason inference/pipeline.py's own docstring gives: the tensor a
served scene carries has to be normalized with the *same* statistics
training used, or this reintroduces the exact train/serve mismatch the
VV_VH_ratio bug already cost real debugging time to find (see
VarunaNet_Spec.md section 14).

Usage:
    python -m inference.stage_demo_scene India_900498 --out demo_scenes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from data.chip_terrain import get_terrain
from data.normalization import NormalizationStats, apply_normalization
from data.sen1floods11 import Sen1Floods11Dataset

DATA_ROOT = Path("datasets/sen1floods11")
STATS_PATH = Path("data/sen1floods11_normalization_stats.json")


def stage_chip(chip_id: str, out_dir: Path) -> Path:
    image_filename = f"{chip_id}_S1Hand.tif"
    dataset = Sen1Floods11Dataset.from_pairs(
        DATA_ROOT / "S1Hand",
        DATA_ROOT / "LabelHand",
        # No label is needed to stage a scene for inference -- the label
        # filename is required by Sen1Floods11Sample's shape but unused
        # here, so a chip's own label file is passed just to satisfy it.
        [(image_filename, f"{chip_id}_LabelHand.tif")],
    )
    sample = dataset[0]

    slope, hand = get_terrain(chip_id, DATA_ROOT / "DEMHand", None)
    full = np.concatenate([sample.image, slope[np.newaxis], hand[np.newaxis]], axis=0)

    stats = NormalizationStats.load(STATS_PATH)
    full = apply_normalization(full, stats)
    full = np.nan_to_num(full, nan=0.0).astype(np.float32)

    # The chip's own georeferencing, read directly from the source file --
    # a served scene must carry real geodesic coordinates or
    # inference/vectorize.py's area computation silently reports square
    # degrees instead of hectares (the exact bug inference/README.md
    # documents finding on the first real-chip run).
    with rasterio.open(DATA_ROOT / "S1Hand" / image_filename) as src:
        transform, crs = src.transform, src.crs

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{chip_id}.tif"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=full.shape[1],
        width=full.shape[2],
        count=full.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(full)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chip_id", help="e.g. India_900498")
    parser.add_argument("--out", default="demo_scenes", type=Path)
    args = parser.parse_args()

    out_path = stage_chip(args.chip_id, args.out)
    print(f"staged {out_path}")


if __name__ == "__main__":
    main()
