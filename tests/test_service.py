"""
Tests for inference/service.py.

A stub session and synthetic districts are injected, so these run without
an ONNX file or the Assam boundaries. What is being tested is the HTTP
surface -- status codes, validation, the readiness distinction, and the
path-traversal guard -- not the segmentation, which has its own tests.
"""

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.crs import CRS

from inference.service import ServiceState, create_app

GEO_TRANSFORM = Affine(1e-4, 0.0, 91.5, 0.0, -1e-4, 26.3)
WGS84 = CRS.from_epsg(4326)


class _WaterSession:
    """Predicts water across the upper half of each tile."""

    def get_inputs(self):
        class _Spec:
            name = "input"

        return [_Spec()]

    def run(self, _outputs, feed):
        n, _, h, w = feed["input"].shape
        logits = np.full((n, 1, h, w), -5.0, dtype=np.float32)
        logits[:, :, : h // 2, :] = 5.0
        return [logits]


def _districts():
    from shapely.geometry import box

    return [
        ("West", box(91.4, 26.2, 91.52, 26.31)),
        ("East", box(91.52, 26.2, 91.7, 26.31)),
    ]


def _stage_scene(root, scene_id="S1A_TEST", height=100, width=130, channels=5):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{scene_id}.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=channels,
        dtype="float32", crs=WGS84, transform=GEO_TRANSFORM,
    ) as dst:
        dst.write(np.zeros((channels, height, width), dtype=np.float32))
    return path


@pytest.fixture
def client(tmp_path):
    _stage_scene(tmp_path / "scenes")
    state = ServiceState(
        session=_WaterSession(),
        model_path="/models/unetpp.int8.onnx",
        scene_root=tmp_path / "scenes",
        districts=_districts(),
    )
    return TestClient(create_app(state))


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model"] == "unetpp.int8.onnx"


def test_health_distinguishes_running_from_ready(tmp_path):
    """
    A health check that returns ok from a service with no model keeps a
    broken instance in the load balancer, which is worse than none.
    """
    client = TestClient(create_app(ServiceState(scene_root=tmp_path)))

    assert client.get("/health").json()["status"] == "no_model"


def test_predict_returns_district_statistics(client):
    # tile_size/overlap sized for the small staged test scene; the 512
    # default would (correctly) reject a 100x130 scene as sub-tile.
    response = client.post(
        "/predict",
        json={"scene_id": "S1A_TEST", "min_area_m2": 0, "tile_size": 40, "overlap": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "S1A_TEST"
    assert body["districts_total"] == 2
    assert body["total_flooded_hectares"] > 0
    assert "processed_at" in body


def test_geometry_is_omitted_unless_asked_for(client):
    """
    One chip's GeoJSON is already ~150KB and a full scene's is far larger,
    so every caller should not pay for it by default.
    """
    small = {"scene_id": "S1A_TEST", "tile_size": 40, "overlap": 10, "min_area_m2": 0}
    without = client.post("/predict", json=small).json()
    with_geom = client.post("/predict", json={**small, "include_geometry": True}).json()

    assert "flood_geojson" not in without
    assert with_geom["flood_geojson"]["type"] == "FeatureCollection"


def test_unknown_scene_is_404_not_500(client):
    assert client.post("/predict", json={"scene_id": "nope"}).status_code == 404


def test_path_traversal_in_scene_id_is_rejected(client):
    """
    scene_id arrives over HTTP, so this request will eventually be made.
    """
    response = client.post("/predict", json={"scene_id": "../../../etc/passwd"})

    assert response.status_code in (400, 404)
    assert "passwd" not in response.text


def test_missing_model_is_503_not_500(tmp_path):
    """
    Correctly built but not ready is retryable for the caller, which a 500
    does not communicate.
    """
    _stage_scene(tmp_path / "scenes")
    client = TestClient(create_app(ServiceState(scene_root=tmp_path / "scenes")))

    assert client.post("/predict", json={"scene_id": "S1A_TEST"}).status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"scene_id": ""},
        {"scene_id": "S1A_TEST", "threshold": 0.0},
        {"scene_id": "S1A_TEST", "threshold": 1.0},
        {"scene_id": "S1A_TEST", "min_area_m2": -1},
        {"scene_id": "S1A_TEST", "tile_size": 0},
    ],
)
def test_invalid_request_bodies_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_overlap_not_smaller_than_tile_is_rejected(client):
    response = client.post(
        "/predict", json={"scene_id": "S1A_TEST", "tile_size": 40, "overlap": 40}
    )

    assert response.status_code == 422
    assert "overlap" in response.text


def test_scene_smaller_than_a_tile_is_a_422_not_a_crash(client, tmp_path):
    _stage_scene(tmp_path / "scenes", scene_id="TINY", height=10, width=10)

    response = client.post(
        "/predict", json={"scene_id": "TINY", "tile_size": 40, "overlap": 10}
    )

    assert response.status_code == 422
