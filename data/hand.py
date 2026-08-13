"""
HAND (Height Above Nearest Drainage), the second terrain-derived channel
in the data contract.

The spec calls this out as "the single most effective auxiliary feature
for SAR flood mapping" -- and the physical reasoning is simple: water
flows downhill and pools in low-lying areas near rivers and streams. A
pixel that is only 1m above the nearest drainage channel is a plausible
flood pixel; a pixel that is 200m above the nearest drainage channel is
not, no matter what its radar backscatter looks like. HAND encodes that
prior directly, instead of asking the model to infer it purely from
SAR intensity.

Unlike slope (a per-pixel formula), HAND requires actual flow-routing:
figuring out which way water would flow off of every pixel, accumulating
that into a drainage network, and then measuring each pixel's elevation
above the nearest point on that network *along the flow path* (not just
"nearest drainage pixel in a straight line"). That's a real hydrology
algorithm with known edge cases (pits, flat areas), so this uses pysheds
-- an established library for exactly this -- rather than reimplementing
D8 flow routing by hand.
"""

import numpy as np

# pysheds 0.5 (the latest release on PyPI) still calls the now-removed
# np.in1d, which NumPy dropped in favor of np.isin. This restores it under
# its old name so pysheds keeps working without pinning numpy to an old
# version project-wide. Safe to remove once pysheds ships a fixed release.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from affine import Affine  # noqa: E402
from pysheds.grid import Grid  # noqa: E402
from pysheds.sview import Raster, ViewFinder  # noqa: E402


def compute_hand(
    dem: np.ndarray, pixel_size_m: float, accumulation_threshold: int = 100
) -> np.ndarray:
    """
    Compute Height Above Nearest Drainage from a DEM patch.

    `accumulation_threshold` is the number of upstream cells that must
    drain through a pixel before it counts as part of the drainage
    network. There's no universally correct value -- it depends on the
    DEM's resolution and the terrain -- so this is a starting default to
    be tuned once real DEM tiles are in use, not a physical constant.

    Note: the outer border of the returned array is NaN. This is a known
    edge effect of flow-routing: border pixels have no upstream neighbors
    inside the patch, so their flow path can't be resolved from this patch
    alone. It goes away once real DEM tiles are fetched with a padding
    buffer around each chip (a later task), so it's left as NaN here
    rather than papering over it with a made-up value.
    """
    if dem.ndim != 2:
        raise ValueError(f"expected a 2D DEM array, got shape {dem.shape}")

    # pysheds' pixel size is signed: positive going right (x), negative
    # going down (y), which is the standard raster convention (north-up).
    affine = Affine(pixel_size_m, 0, 0, 0, -pixel_size_m, 0)
    viewfinder = ViewFinder(affine=affine, shape=dem.shape, nodata=np.nan)
    dem_raster = Raster(dem.astype(np.float64), viewfinder=viewfinder)

    grid = Grid.from_raster(dem_raster)

    # Flow routing assumes every pixel has somewhere lower to drain to.
    # Real DEMs have noise-induced pits and flat plateaus that break that
    # assumption, so these three steps condition the DEM before routing.
    pit_filled = grid.fill_pits(dem_raster)
    depression_filled = grid.fill_depressions(pit_filled)
    conditioned = grid.resolve_flats(depression_filled)

    flow_direction = grid.flowdir(conditioned)
    accumulation = grid.accumulation(flow_direction)
    drainage_mask = accumulation > accumulation_threshold

    hand = grid.compute_hand(flow_direction, conditioned, drainage_mask)
    return np.asarray(hand, dtype=np.float32)
