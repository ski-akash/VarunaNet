"""
The FastAPI service that fronts the inference pipeline.

This is what the Node gateway calls. It is deliberately thin: it owns the
model session, validates what comes in, and delegates to
inference/pipeline.py. Anything that could be a pure function lives in the
modules it calls, so the interesting logic stays testable without HTTP.

**The model is loaded once, at startup, not per request.** An ONNX session
takes seconds to construct -- graph load, optimisation, arena allocation --
and doing that per request would dominate a latency budget where the
actual forward pass is ~700ms per tile. It also means a broken or missing
model file fails at boot, loudly, rather than on the first user request.

**Scenes are not accepted over HTTP.** A real Sentinel-1 scene is ~8GB as
float32; posting one would be absurd. The service takes a *reference* to a
scene the worker has already staged (spec section 5's ingest pipeline puts
it there) and reads it locally. The request body carries the scene id and
options, never pixels.

**Predictions are not cached here.** The Redis semantic/aggregate cache
(spec section 5) sits in the gateway, in front of this service, because
what is worth caching is the district-level answer, not a raster.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.pipeline import run_pipeline_on_mask
from inference.streaming import predict_scene_streaming
from inference.tiling import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE
from inference.vectorize import DEFAULT_MIN_AREA_M2

# Where the worker stages scenes for this service to read. Configurable
# because the container mounts it somewhere different from a dev machine.
SCENE_ROOT_ENV = "VARUNANET_SCENE_ROOT"
MODEL_PATH_ENV = "VARUNANET_MODEL_PATH"


class PredictRequest(BaseModel):
    scene_id: str = Field(..., min_length=1, description="Staged scene identifier")
    threshold: float = Field(0.5, gt=0.0, lt=1.0)
    min_area_m2: float = Field(DEFAULT_MIN_AREA_M2, ge=0.0)
    tile_size: int = Field(DEFAULT_TILE_SIZE, gt=0)
    overlap: int = Field(DEFAULT_OVERLAP, ge=0)
    include_geometry: bool = Field(
        False,
        description=(
            "Return the flood polygons as well as the district summary. Off by "
            "default: the GeoJSON for one chip is already ~150KB and a full "
            "scene's is far larger, so the map fetches it separately rather "
            "than every caller paying for it."
        ),
    )


@dataclass
class ServiceState:
    """Held on the app rather than in module globals, so tests can inject."""

    session: object | None = None
    model_path: str = "unset"
    scene_root: Path = Path(".")
    districts: list | None = None


def load_session(model_path: str):
    """Construct the ONNX session. Separated so tests can skip it."""
    import onnxruntime

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"model not found at {model_path}. Set {MODEL_PATH_ENV} to an "
            "exported .onnx (see inference/export_onnx.py)."
        )
    return onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )


def create_app(state: Optional[ServiceState] = None) -> FastAPI:
    """
    Build the app. `state` is injectable so tests can supply a stub session
    and synthetic districts without an ONNX file or the Assam boundaries.
    """
    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        # Model load happens here rather than per request: an ONNX session
        # takes seconds to build, and a missing or broken model file should
        # fail at boot, loudly, not on a user's first request.
        svc: ServiceState = instance.state.svc
        if svc.session is None:  # None means "not injected by a caller/test"
            model_path = os.environ.get(MODEL_PATH_ENV)
            if model_path:
                svc.session = load_session(model_path)
                svc.model_path = model_path
            svc.scene_root = Path(os.environ.get(SCENE_ROOT_ENV, "."))
        yield

    app = FastAPI(
        title="VarunaNet inference",
        description="Tiled SAR flood segmentation with district-level aggregation",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.svc = state or ServiceState()

    @app.get("/health")
    def health() -> dict:
        """
        Reports whether a model is actually loaded, not just that the
        process is up. A health check that returns 200 from a service with
        no model is worse than no health check -- it keeps a broken
        instance in the load balancer.
        """
        svc: ServiceState = app.state.svc
        return {
            "status": "ok" if svc.session is not None else "no_model",
            "model": Path(svc.model_path).name if svc.session else None,
            "scene_root": str(svc.scene_root),
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict:
        svc: ServiceState = app.state.svc
        if svc.session is None:
            # 503, not 500: the service is correctly built but not ready,
            # which is a retryable condition for the caller.
            raise HTTPException(status_code=503, detail="no model loaded")
        if request.overlap >= request.tile_size:
            raise HTTPException(
                status_code=422,
                detail=f"overlap {request.overlap} must be less than tile_size "
                f"{request.tile_size}",
            )

        scene_path = _resolve_scene(svc.scene_root, request.scene_id)
        scene, transform, crs = _read_scene(scene_path)

        height, width = scene.shape[1], scene.shape[2]
        mask = np.zeros((height, width), dtype=bool)

        def write_band(first_row: int, band: np.ndarray) -> None:
            mask[first_row : first_row + band.shape[0]] = band

        def read_window(row: int, col: int, size: int) -> np.ndarray:
            return scene[:, row : row + size, col : col + size]

        try:
            predict_scene_streaming(
                svc.session,
                read_window,
                height,
                width,
                write_band,
                tile_size=request.tile_size,
                overlap=request.overlap,
                threshold=request.threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result = run_pipeline_on_mask(
            mask,
            transform,
            crs,
            scene_id=request.scene_id,
            model_path=svc.model_path,
            districts=svc.districts,
            min_area_m2=request.min_area_m2,
        )

        payload = result.summary()
        if request.include_geometry:
            payload["flood_geojson"] = result.flood_geojson
        return payload

    return app


def _resolve_scene(scene_root: Path, scene_id: str) -> Path:
    """
    Map a scene id to a staged file, refusing anything that escapes the
    scene root.

    scene_id arrives over HTTP, so "../../etc/passwd" is a request that
    will eventually be made. Resolving both sides and checking containment
    is the check that actually holds, rather than blacklisting "..".
    """
    root = scene_root.resolve()
    candidate = (root / f"{scene_id}.tif").resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=400, detail="invalid scene_id")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"scene {scene_id} not staged")
    return candidate


def _read_scene(path: Path):
    """Read a staged scene as (channels, height, width) plus its georeferencing."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.transform, src.crs


# Module-level app for `uvicorn inference.service:app`.
app = create_app()
