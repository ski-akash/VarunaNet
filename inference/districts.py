"""
Aggregates flood polygons to districts -- the numbers the dashboard and
the chat panel actually answer questions with.

Everything upstream produces geometry: a mask becomes polygons in
inference/vectorize.py. Nobody asks a question in those terms. They ask
"which districts are worst hit this week", and that needs flooded area
per district, as a share of that district's own area, ranked.

This is deliberately pure geometry with no database in it. The same
computation will eventually run as a PostGIS spatial join for scenes at
rest, but keeping a self-contained implementation means the numbers can be
produced and tested without standing up a database, and gives the PostGIS
version something to be checked against.

**Share of district, not raw area, is what ranks meaningfully.** Assam's
districts differ in size by more than an order of magnitude, so ranking by
flooded km2 mostly ranks districts by how large they are. A small district
half under water is the emergency; a large one with the same absolute
flooded area may be barely affected. Both numbers are reported, and
`worst_affected` sorts on the fraction.

Districts are clipped against the flood geometry with a real intersection
rather than a centroid or bounding-box test: floods follow river courses
and routinely straddle several districts, so anything coarser would assign
a whole flood to one district and report zero for its neighbours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Assam's 27 districts, as extracted for the frontend map. Reusing the
# same file means the dashboard's boundaries and the reported statistics
# can never disagree about where a district is.
DEFAULT_DISTRICTS_PATH = Path("frontend/public/geo/assam_districts.geojson")


@dataclass
class DistrictImpact:
    name: str
    district_area_m2: float
    flooded_area_m2: float

    @property
    def flooded_fraction(self) -> float:
        if self.district_area_m2 <= 0:
            return 0.0
        return self.flooded_area_m2 / self.district_area_m2

    @property
    def flooded_percent(self) -> float:
        return self.flooded_fraction * 100.0

    @property
    def flooded_hectares(self) -> float:
        return self.flooded_area_m2 / 10_000.0


def load_districts(path: Path = DEFAULT_DISTRICTS_PATH) -> list[tuple[str, object]]:
    """
    (name, geometry) for each district, geometry as a shapely object.

    Raises rather than returning empty if the file is missing: silently
    reporting "no districts affected" because a path was wrong is the
    failure mode this whole module exists to make impossible.
    """
    from shapely.geometry import shape

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"district boundaries not found at {path}. Without them every "
            "district would report zero flooding, which is indistinguishable "
            "from a genuinely dry week."
        )

    collection = json.loads(path.read_text())
    districts = []
    for feature in collection["features"]:
        name = feature["properties"].get("name")
        if not name:
            continue
        districts.append((name, shape(feature["geometry"])))

    if not districts:
        raise ValueError(f"{path} contained no named district features")
    return districts


def _geodesic_area_m2(geometry) -> float:
    """
    Area in square metres for geometry in EPSG:4326.

    The district boundaries and the flood polygons are both in degrees, so
    a planar area would be in square degrees -- the same units bug that
    made a fully flooded scene report zero hectares in
    inference/vectorize.py. Computed on the ellipsoid instead.
    """
    from pyproj import Geod

    if geometry.is_empty:
        return 0.0
    area, _perimeter = Geod(ellps="WGS84").geometry_area_perimeter(geometry)
    return abs(float(area))


def aggregate_to_districts(
    flood_geojson: dict,
    districts: list[tuple[str, object]] | None = None,
    districts_path: Path = DEFAULT_DISTRICTS_PATH,
) -> list[DistrictImpact]:
    """
    Flooded area per district, for every district (including untouched
    ones, reported as zero).

    Untouched districts are included on purpose. "Dhemaji is not flooded"
    is a real answer to a real question, and a list that silently omits
    dry districts cannot distinguish them from districts that were never
    covered by the scene at all.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if districts is None:
        districts = load_districts(districts_path)

    features = flood_geojson.get("features", [])
    flood_geometries = [shape(f["geometry"]) for f in features]

    # Union first: overlapping flood polygons would otherwise have their
    # overlap counted twice per district, inflating the total.
    flood = unary_union(flood_geometries) if flood_geometries else None

    impacts = []
    for name, boundary in districts:
        district_area = _geodesic_area_m2(boundary)
        if flood is None or flood.is_empty:
            flooded_area = 0.0
        else:
            # Cheap rejection first: most districts don't touch most
            # floods, and intersects() is far cheaper than intersection().
            flooded_area = (
                _geodesic_area_m2(flood.intersection(boundary))
                if flood.intersects(boundary)
                else 0.0
            )
        impacts.append(
            DistrictImpact(
                name=name,
                district_area_m2=district_area,
                flooded_area_m2=flooded_area,
            )
        )
    return impacts


def worst_affected(
    impacts: list[DistrictImpact], limit: int | None = None
) -> list[DistrictImpact]:
    """
    Districts ranked by the share of their own area under water.

    Ranked on fraction rather than absolute area because Assam's districts
    vary in size by more than 10x, so ranking by km2 largely reproduces a
    ranking of district size. Ties broken by absolute area so the ordering
    is deterministic.
    """
    ranked = sorted(
        impacts,
        key=lambda i: (i.flooded_fraction, i.flooded_area_m2),
        reverse=True,
    )
    return ranked[:limit] if limit is not None else ranked


def summarize(impacts: list[DistrictImpact]) -> dict:
    """
    A compact, JSON-serialisable summary -- the shape an API response or a
    grounded LLM tool call would return.

    Every number here traces to a computed value rather than a model's
    recollection, which is the point: spec section 6.1's rule is that the
    LLM may never state a figure it did not get from a tool call.
    """
    affected = [i for i in impacts if i.flooded_area_m2 > 0]
    return {
        "districts_total": len(impacts),
        "districts_affected": len(affected),
        "total_flooded_hectares": round(
            sum(i.flooded_hectares for i in impacts), 1
        ),
        "worst_affected": [
            {
                "name": i.name,
                "flooded_hectares": round(i.flooded_hectares, 1),
                "flooded_percent": round(i.flooded_percent, 2),
            }
            for i in worst_affected(affected, limit=5)
        ],
    }
