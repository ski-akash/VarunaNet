"""
Terrain-derived auxiliary layers, starting with slope.

Slope matters here because of the "look-alike" false positive problem
named in the project spec: radar shadow (the dark patch cast behind a hill,
out of the radar's line of sight) looks exactly like water in SAR -- both
scatter the radar pulse away from the sensor instead of back to it. But
radar shadow only happens on steep terrain, and water doesn't sit on steep
terrain (it flows downhill and pools flat). A slope layer lets the model
tell "this is a hillside in shadow" apart from "this is actually flooded",
cheaply, without having to learn that distinction purely from backscatter
values.

HAND (Height Above Nearest Drainage), the other terrain-derived channel in
the data contract, needs a full flow-routing algorithm rather than a local
pixel formula, so it lives in its own module (added separately) instead of
here.
"""

import numpy as np


def compute_slope(dem: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """
    Compute terrain slope, in degrees, from a DEM patch.

    `pixel_size_m` is the ground distance one pixel covers (e.g. 30 for a
    30m-resolution DEM). It's needed to turn an elevation change in meters
    between neighboring pixels into an actual slope *angle*, rather than a
    unit-less "difference between neighbors" number that would mean
    something different for a 10m DEM than for a 30m one.
    """
    if dem.ndim != 2:
        raise ValueError(f"expected a 2D DEM array, got shape {dem.shape}")

    # np.gradient gives the rate of elevation change per pixel along each
    # axis (rise / run, in meters per pixel) once divided by the real-world
    # pixel size.
    dz_dy, dz_dx = np.gradient(dem.astype(np.float64), pixel_size_m)

    # Combine the two directional gradients into a single rise/run ratio,
    # then convert that ratio into a slope angle in degrees.
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    return np.degrees(np.arctan(rise_run)).astype(np.float32)
