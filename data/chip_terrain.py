"""
Convenience helpers for turning one Sen1Floods11 chip's DEM into both
terrain-derived data-contract channels together (slope, HAND), and for
loading that DEM chip in the first place.

Split out from data/terrain.py (slope) and data/hand.py (HAND) themselves,
which are deliberately about the terrain *algorithms* in isolation,
testable against any DEM array -- this module is specifically about the
Sen1Floods11 *chip* file convention (a co-registered *_DEMHand.tif per
chip) and this project's specific choice of pixel size and HAND
accumulation threshold.

Originally lived only inside benchmarks/evaluate.py, duplicated by
training/sen1floods11_dataset.py when the real CNN training path needed
the exact same "chip DEM -> slope + HAND" step. Moved here instead of
just importing one from the other so both consumers read from one place
-- the same reasoning data/contract.py itself is built on: two copies of
"how do you turn a DEM into a data-contract channel for this project"
would eventually drift, and that drift would show up as a silently wrong
flood map, not a loud error.
"""

import time
from pathlib import Path

import numpy as np
import rasterio

from data.hand import compute_hand
from data.terrain import compute_slope

# Sen1Floods11 chips are stored in EPSG:4326 (lat/lon degrees), not a
# projected metric CRS, so there's no exact "meters per pixel" to read
# from the file. 10m is the native ground sampling distance of the
# underlying Sentinel-1 product; treating it as isotropic here is an
# approximation (true east-west ground distance shrinks slightly with
# latitude) -- the same not-a-tuned-constant spirit as the accumulation
# threshold below.
CHIP_PIXEL_SIZE_M = 10.0

# Genuinely terrain-dependent, not a universal constant -- see
# data/hand.py's compute_hand docstring. This is the same default already
# validated against real chips for the Random Forest baseline.
HAND_ACCUMULATION_THRESHOLD = 100


def load_dem(dem_path: str | Path) -> np.ndarray:
    """Read a single-band *_DEMHand.tif chip, already co-registered to its S1 chip's grid."""
    with rasterio.open(dem_path) as src:
        return src.read(1).astype(np.float32)


def compute_terrain_layers(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive slope and HAND from a chip's DEM, on the chip's own pixel grid."""
    slope = compute_slope(dem, CHIP_PIXEL_SIZE_M)
    hand = compute_hand(dem, CHIP_PIXEL_SIZE_M, accumulation_threshold=HAND_ACCUMULATION_THRESHOLD)
    return slope, hand


# Keyed by chip id, so the same chip's slope/HAND can be computed once and
# reused everywhere that chip shows up, instead of recomputing pysheds flow
# routing from scratch each time. This matters most for two callers, both
# of which touch the same chip repeatedly: hold-one-event-out CV
# (benchmarks/hold_one_event_out.py -- most chips are part of the training
# set in most of the 11 folds) and real CNN training
# (training/sen1floods11_dataset.py -- the *same* 252 training chips get
# read again every single epoch, so recomputing HAND on the fly per
# __getitem__ call would make pysheds flow-routing the dominant cost of
# training instead of the model itself).
TerrainCache = dict[str, tuple[np.ndarray, np.ndarray]]


def get_terrain(
    chip_id: str, dem_dir: str | Path, terrain_cache: TerrainCache | None
) -> tuple[np.ndarray, np.ndarray]:
    """Look up a chip's terrain in `terrain_cache` if present, else compute it directly."""
    if terrain_cache is not None and chip_id in terrain_cache:
        return terrain_cache[chip_id]
    dem = load_dem(Path(dem_dir) / f"{chip_id}_DEMHand.tif")
    return compute_terrain_layers(dem)


def build_terrain_cache(chip_ids: list[str], dem_dir: str | Path) -> TerrainCache:
    """
    Compute slope+HAND once per chip id and return them as a cache, ready
    to hand to any caller that reads the same chips repeatedly (see
    TerrainCache's comment above for who and why).

    Flow-routing time varies a lot chip to chip -- a very flat DEM (e.g. a
    river delta) can make pysheds' flat-resolution step run far longer than
    a typical hilly chip, so this logs progress and per-chip timing rather
    than running silently for however long the slowest chips take.
    """
    dem_dir = Path(dem_dir)
    cache: TerrainCache = {}
    for i, chip_id in enumerate(chip_ids, start=1):
        start = time.monotonic()
        dem = load_dem(dem_dir / f"{chip_id}_DEMHand.tif")
        cache[chip_id] = compute_terrain_layers(dem)
        elapsed = time.monotonic() - start
        print(f"[terrain {i}/{len(chip_ids)}] {chip_id} ({elapsed:.1f}s)")
    return cache
