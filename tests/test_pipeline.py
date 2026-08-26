"""
Tests for inference/pipeline.py -- the seam between the stages.

The stages themselves are tested in their own files. What matters here is
that the transform and CRS survive the whole way through, that the summary
payload is complete and serialisable, and that the pipeline agrees with
the streaming path, since those are the joins where a mistake produces
plausible output rather than an error.
"""

import json

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from inference.pipeline import run_pipeline, run_pipeline_on_mask
from inference.streaming import array_reader, collect_bands, predict_scene_streaming

# ~0.0001 degree pixels over Assam's Brahmaputra valley.
GEO_TRANSFORM = Affine(1e-4, 0.0, 91.5, 0.0, -1e-4, 26.3)
WGS84 = CRS.from_epsg(4326)
TILE = 40
OVERLAP = 10


def _districts():
    """Two synthetic districts straddling the scene."""
    from shapely.geometry import box

    return [
        ("West", box(91.4, 26.2, 91.52, 26.31)),
        ("East", box(91.52, 26.2, 91.7, 26.31)),
    ]


class _WaterBandSession:
    """Predicts water in the upper half of every tile."""

    def get_inputs(self):
        class _Spec:
            name = "input"

        return [_Spec()]

    def run(self, _outputs, feed):
        batch = feed["input"]
        n, _, h, w = batch.shape
        logits = np.full((n, 1, h, w), -5.0, dtype=np.float32)
        logits[:, :, : h // 2, :] = 5.0
        return [logits]


def _scene(height=100, width=130):
    return np.zeros((5, height, width), dtype=np.float32)


def test_pipeline_produces_polygons_and_district_stats():
    result = run_pipeline(
        _WaterBandSession(),
        _scene(),
        GEO_TRANSFORM,
        WGS84,
        scene_id="S1A_TEST_0001",
        model_path="/models/unetpp.int8.onnx",
        districts=_districts(),
        tile_size=TILE,
        overlap=OVERLAP,
        min_area_m2=0.0,
    )

    assert result.scene_id == "S1A_TEST_0001"
    assert result.vectorized.polygons, "a predicted flood produced no polygons"
    assert len(result.districts) == 2
    assert any(d.flooded_area_m2 > 0 for d in result.districts)


def test_areas_are_real_not_square_degrees():
    """
    The regression the whole pipeline is most exposed to: if the CRS does
    not survive to vectorize_mask, every area is in square degrees and the
    scene reports zero flooding without raising.
    """
    result = run_pipeline(
        _WaterBandSession(),
        _scene(),
        GEO_TRANSFORM,
        WGS84,
        scene_id="S1A_TEST_0002",
        districts=_districts(),
        tile_size=TILE,
        overlap=OVERLAP,
    )

    # Upper half of a 100x130 scene of 1e-4 deg pixels is ~11m x 100 x 130/2
    # -> on the order of tens of hectares, definitely not zero.
    assert result.vectorized.total_area_hectares > 1.0


def test_summary_is_complete_and_json_serialisable():
    result = run_pipeline(
        _WaterBandSession(),
        _scene(),
        GEO_TRANSFORM,
        WGS84,
        scene_id="S1A_TEST_0003",
        model_path="/models/unetpp.int8.onnx",
        districts=_districts(),
        tile_size=TILE,
        overlap=OVERLAP,
        min_area_m2=0.0,
    )

    summary = result.summary()
    json.dumps(summary)  # must not raise

    # Provenance travels with the figures, so a stated number is checkable.
    assert summary["scene_id"] == "S1A_TEST_0003"
    assert summary["model"] == "unetpp.int8.onnx"
    assert summary["processed_at"].endswith("+00:00"), "timestamp must be UTC-aware"
    for key in (
        "water_pixel_fraction",
        "flood_polygons",
        "districts_total",
        "districts_affected",
        "total_flooded_hectares",
        "worst_affected",
    ):
        assert key in summary


def test_pipeline_matches_the_streaming_path():
    """
    A real scene runs through inference.streaming, not the in-memory
    tiler. Both must produce the same district numbers, or the answer
    would depend on which path a scene happened to take.
    """
    session = _WaterBandSession()
    scene = _scene()
    height, width = scene.shape[1], scene.shape[2]

    in_memory = run_pipeline(
        session, scene, GEO_TRANSFORM, WGS84, "A",
        districts=_districts(), tile_size=TILE, overlap=OVERLAP, min_area_m2=0.0,
    )

    mask, writer = collect_bands(height, width)
    predict_scene_streaming(
        session, array_reader(scene), height, width, writer,
        tile_size=TILE, overlap=OVERLAP,
    )
    streamed = run_pipeline_on_mask(
        mask, GEO_TRANSFORM, WGS84, "A", districts=_districts(), min_area_m2=0.0
    )

    assert streamed.water_pixel_fraction == pytest.approx(
        in_memory.water_pixel_fraction
    )
    assert streamed.vectorized.total_area_m2 == pytest.approx(
        in_memory.vectorized.total_area_m2
    )
    assert [d.flooded_area_m2 for d in streamed.districts] == pytest.approx(
        [d.flooded_area_m2 for d in in_memory.districts]
    )


def test_a_dry_scene_reports_zero_rather_than_failing():
    class _DrySession(_WaterBandSession):
        def run(self, _outputs, feed):
            n, _, h, w = feed["input"].shape
            return [np.full((n, 1, h, w), -5.0, dtype=np.float32)]

    result = run_pipeline(
        _DrySession(), _scene(), GEO_TRANSFORM, WGS84, "dry",
        districts=_districts(), tile_size=TILE, overlap=OVERLAP,
    )

    assert result.vectorized.polygons == []
    assert result.summary()["districts_affected"] == 0
    assert result.summary()["total_flooded_hectares"] == 0.0


def test_rejects_a_scene_that_is_not_channel_first():
    with pytest.raises(ValueError, match="channels, height, width"):
        run_pipeline(
            _WaterBandSession(),
            np.zeros((100, 130), dtype=np.float32),
            GEO_TRANSFORM,
            WGS84,
            "bad",
            districts=_districts(),
            tile_size=TILE,
        )
