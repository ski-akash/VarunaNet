"""
Turns a pixel flood mask into vector polygons, which is what the database
and the map actually consume.

The dashboard never queries pixels. It asks questions like "how much of
Nagaon is flooded this week", and answering that against a raster means
rasterizing district boundaries and counting -- per district, per query.
As polygons in PostGIS it is one spatial join, and the geometry can be
served to MapLibre directly.

Three things happen here, and the middle one is the reason this module is
not a one-line call to `rasterio.features.shapes`:

1. **Trace the mask into polygons**, in real-world coordinates via the
   scene's affine transform, so the output is georeferenced rather than in
   pixel space.

2. **Drop specks.** SAR is speckle-noisy and a segmentation mask of a real
   scene contains thousands of isolated one- and two-pixel blobs. Left in,
   they dominate the polygon count, bloat the payload the browser has to
   parse, and slow every spatial join -- while representing nothing anyone
   would call a flood. They are filtered by real area, not pixel count, so
   the threshold means the same thing at any resolution.

3. **Simplify.** Traced polygons follow pixel edges exactly, so every
   boundary is a staircase with a vertex every 10m. Simplifying with a
   tolerance tied to the pixel size removes the staircase without moving
   the boundary meaningfully -- the same reasoning as the mapshaper pass
   on the district boundaries in the frontend, where an unsimplified
   outline was both slower and no more accurate.

Holes are preserved. A dry patch inside a flooded area is real
information, and dropping interiors would systematically overstate flood
extent -- the direction of error that matters most here.

**Areas are geodesic when the scene is in a geographic CRS.** Sen1Floods11
chips are EPSG:4326, so the affine's units are *degrees*, and taking its
determinant as an area gives square degrees -- around 8e-09 per pixel.
Every polygon then falls under any sane minimum-area threshold and the
whole scene silently reports zero flooding. That is exactly what happened
on the first real chip run here: a scene 77.6% covered in predicted water
produced 0.0 hectares and 14 regions all discarded as specks. So the CRS
is inspected, and geographic coordinates get true ellipsoidal areas via
pyproj's Geod rather than a units-blind determinant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from affine import Affine

# Polygons smaller than this are speckle, not flood. 10,000 m2 is one
# hectare, which at Sentinel-1's 10m pixels is 100 pixels -- comfortably
# above the isolated-blob range while far below anything that would matter
# for district-level reporting.
DEFAULT_MIN_AREA_M2 = 10_000.0

# Simplification tolerance as a fraction of one pixel. Half a pixel keeps
# the boundary within sub-pixel distance of the traced original while
# removing essentially all of the staircase.
DEFAULT_SIMPLIFY_PIXEL_FRACTION = 0.5


@dataclass
class FloodPolygon:
    """One flooded region, in scene coordinates."""

    geometry: dict  # GeoJSON geometry (Polygon), ready for PostGIS/MapLibre
    area_m2: float

    @property
    def area_hectares(self) -> float:
        return self.area_m2 / 10_000.0


@dataclass
class VectorizeResult:
    polygons: list[FloodPolygon]
    total_area_m2: float
    traced_count: int
    dropped_as_speck: int

    @property
    def total_area_hectares(self) -> float:
        return self.total_area_m2 / 10_000.0


def _pixel_area_native(transform: Affine) -> float:
    """
    Area of one pixel in the transform's own units (m2, or deg2 for a
    geographic CRS).

    abs(a * e - b * d) is the determinant of the affine's linear part --
    the correct area scale for a rotated or sheared transform, not just an
    axis-aligned one. Sentinel-1 GRD products are usually north-up, but a
    reprojected scene need not be, and using abs(a) * abs(e) would then be
    quietly wrong.
    """
    return abs(transform.a * transform.e - transform.b * transform.d)


def _is_geographic(crs) -> bool:
    """
    True if `crs` measures in degrees rather than a linear unit.

    None means "caller did not say", and the safe assumption there is
    projected/metres: that is what the affine determinant already implies,
    so it keeps the no-CRS path behaving as before rather than silently
    switching to geodesic maths on an unknown reference.
    """
    if crs is None:
        return False
    try:
        return bool(crs.is_geographic)
    except AttributeError:
        from pyproj import CRS

        return bool(CRS.from_user_input(crs).is_geographic)


def _area_m2(geometry, crs, native_pixel_area: float, transform: Affine) -> float:
    """
    True area of `geometry` in square metres.

    Projected CRS: shapely's planar area is already in the CRS's linear
    unit. Geographic CRS: shapely would return square degrees, whose
    metre-equivalent varies with latitude, so the area is computed on the
    ellipsoid instead.
    """
    if not _is_geographic(crs):
        return float(geometry.area)

    from pyproj import Geod

    # abs(): the sign encodes ring orientation, not a negative area.
    area, _perimeter = Geod(ellps="WGS84").geometry_area_perimeter(geometry)
    return abs(float(area))


def vectorize_mask(
    mask: np.ndarray,
    transform: Affine,
    crs=None,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
    simplify_pixel_fraction: float = DEFAULT_SIMPLIFY_PIXEL_FRACTION,
) -> VectorizeResult:
    """
    Trace `mask` into simplified, speck-filtered georeferenced polygons.

    `mask` is boolean, True meaning water. `transform` maps pixel indices
    to world coordinates -- rasterio's `dataset.transform`. `crs` is that
    dataset's CRS; pass it whenever it is known, because without it a
    geographic scene's areas come out in square degrees and every polygon
    is discarded as a speck (see the module docstring).
    """
    from rasterio import features
    from shapely.geometry import mapping, shape

    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    if min_area_m2 < 0:
        raise ValueError(f"min_area_m2 must be non-negative, got {min_area_m2}")

    # Simplification works in the transform's own units, so it uses the
    # native pixel size regardless of CRS -- a half-pixel tolerance means
    # the same thing in degrees as in metres.
    native_pixel_area = _pixel_area_native(transform)
    tolerance = float(np.sqrt(native_pixel_area)) * simplify_pixel_fraction

    polygons: list[FloodPolygon] = []
    traced = 0
    dropped = 0

    # mask=mask restricts tracing to True pixels, so the dry background
    # never becomes one enormous polygon with the flood as its holes.
    for geom, value in features.shapes(
        mask.astype(np.uint8), mask=mask, transform=transform
    ):
        if not value:
            continue
        traced += 1

        geometry = shape(geom)
        if tolerance > 0:
            # preserve_topology stops simplification from producing
            # self-intersecting rings, which PostGIS rejects on insert.
            geometry = geometry.simplify(tolerance, preserve_topology=True)

        # Area is measured after simplification, since that is the geometry
        # actually being stored and reported on.
        area = _area_m2(geometry, crs, native_pixel_area, transform)
        if area < min_area_m2:
            dropped += 1
            continue
        if geometry.is_empty:
            dropped += 1
            continue

        polygons.append(FloodPolygon(geometry=mapping(geometry), area_m2=area))

    polygons.sort(key=lambda p: p.area_m2, reverse=True)
    return VectorizeResult(
        polygons=polygons,
        total_area_m2=float(sum(p.area_m2 for p in polygons)),
        traced_count=traced,
        dropped_as_speck=dropped,
    )


def to_feature_collection(result: VectorizeResult) -> dict:
    """
    A GeoJSON FeatureCollection, the form both PostGIS ingestion and the
    MapLibre source in the frontend already understand.
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": polygon.geometry,
                "properties": {
                    "area_m2": round(polygon.area_m2, 1),
                    "area_hectares": round(polygon.area_hectares, 3),
                },
            }
            for polygon in result.polygons
        ],
    }


def iter_wkt(result: VectorizeResult) -> Iterator[tuple[str, float]]:
    """
    (WKT, area_m2) per polygon, for inserting into a PostGIS geometry
    column via ST_GeomFromText without a GeoJSON round-trip.
    """
    from shapely.geometry import shape

    for polygon in result.polygons:
        yield shape(polygon.geometry).wkt, polygon.area_m2
