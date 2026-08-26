"""
The whole serving path in one call: a scene goes in, district-level flood
statistics come out.

The individual stages each exist for their own reasons and are tested
separately -- export, tiling, streaming, vectorizing, district
aggregation. This module is the seam between them, and the seam is worth
its own module because that is where the units, the coordinate reference,
and the shapes have to agree, and where they have historically not:

- the affine transform has to travel from the raster all the way to
  `vectorize_mask`, or areas come out in square degrees and a flooded
  scene reports zero hectares
- the CRS has to travel with it, for the same reason
- the mask has to be thresholded once, on blended logits, not per tile

Getting any of those wrong produces plausible-looking output rather than
an error, which is why the pipeline is expressed once here instead of
being reassembled by each caller.

`run_pipeline` returns everything downstream needs and nothing it doesn't:
the district table for the dashboard and the LLM tool layer, the polygons
for the map, and enough provenance to trace any number back to the scene
and model that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from affine import Affine

from inference.districts import DistrictImpact, aggregate_to_districts, summarize
from inference.tiling import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE
from inference.vectorize import (
    DEFAULT_MIN_AREA_M2,
    VectorizeResult,
    to_feature_collection,
    vectorize_mask,
)


@dataclass
class PipelineResult:
    """Everything a scene run produces, with the provenance to trace it."""

    scene_id: str
    model_path: str
    processed_at: str
    water_pixel_fraction: float
    vectorized: VectorizeResult
    districts: list[DistrictImpact]
    flood_geojson: dict = field(repr=False)

    def summary(self) -> dict:
        """
        The JSON payload the API returns and the LLM tool layer reads.

        Provenance is included in the same object as the figures on
        purpose: spec section 6.1's rule is that the model may never state
        a number it did not get from a tool call, and a payload that
        carries its own source makes a stated number checkable rather
        than merely plausible.
        """
        return {
            "scene_id": self.scene_id,
            "model": Path(self.model_path).name,
            "processed_at": self.processed_at,
            "water_pixel_fraction": round(self.water_pixel_fraction, 6),
            "flood_polygons": len(self.vectorized.polygons),
            "specks_dropped": self.vectorized.dropped_as_speck,
            **summarize(self.districts),
        }


def run_pipeline(
    session,
    scene: np.ndarray,
    transform: Affine,
    crs,
    scene_id: str,
    model_path: str = "unknown",
    districts=None,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = 8,
    threshold: float = 0.5,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> PipelineResult:
    """
    Run one scene end to end.

    `scene` is (channels, height, width) already normalized to the data
    contract -- this function deliberately does not normalize, because the
    statistics used must be the same ones the model trained with, and
    silently applying a different set is exactly the train/serve mismatch
    that made every pre-fix checkpoint's evaluation meaningless.

    Scenes that fit in memory go through the in-memory tiler; a real
    25,000 x 16,000 scene should use inference.streaming instead, which
    produces byte-identical output without the 14GB peak.
    """
    from inference.tiling import predict_scene

    if scene.ndim != 3:
        raise ValueError(f"expected (channels, height, width), got {scene.shape}")

    mask = predict_scene(
        session,
        scene,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        threshold=threshold,
    )
    return _finish(
        mask=mask,
        transform=transform,
        crs=crs,
        scene_id=scene_id,
        model_path=model_path,
        districts=districts,
        min_area_m2=min_area_m2,
    )


def run_pipeline_on_mask(
    mask: np.ndarray,
    transform: Affine,
    crs,
    scene_id: str,
    model_path: str = "unknown",
    districts=None,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> PipelineResult:
    """
    The half of the pipeline after inference, for a mask produced
    elsewhere -- by the streaming path, or read back from a stored raster
    so a scene can be re-aggregated without re-running the model.
    """
    return _finish(
        mask=mask,
        transform=transform,
        crs=crs,
        scene_id=scene_id,
        model_path=model_path,
        districts=districts,
        min_area_m2=min_area_m2,
    )


def _finish(
    mask: np.ndarray,
    transform: Affine,
    crs,
    scene_id: str,
    model_path: str,
    districts,
    min_area_m2: float,
) -> PipelineResult:
    vectorized = vectorize_mask(mask, transform, crs=crs, min_area_m2=min_area_m2)
    flood_geojson = to_feature_collection(vectorized)
    impacts = aggregate_to_districts(flood_geojson, districts=districts)

    return PipelineResult(
        scene_id=scene_id,
        model_path=model_path,
        # Timezone-aware and in UTC: a naive local timestamp on a stored
        # scene is ambiguous the moment anything runs in another zone.
        processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        water_pixel_fraction=float(mask.mean()),
        vectorized=vectorized,
        districts=impacts,
        flood_geojson=flood_geojson,
    )
