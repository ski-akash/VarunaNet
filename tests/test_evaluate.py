"""
Tests for the evaluation wiring in benchmarks/evaluate.py, using tiny
synthetic Sen1Floods11-shaped fixtures (image + label + DEM GeoTIFFs) the
same way tests/test_sen1floods11.py and tests/test_hand.py do, instead of
the real ~1.1GB downloaded dataset.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import data.chip_terrain as chip_terrain_module
from benchmarks.evaluate import (
    build_terrain_cache,
    compute_terrain_layers,
    evaluate_baseline,
    load_dem,
    otsu_hand_predict,
    otsu_predict,
    train_random_forest_baseline,
)
from data.contract import LABEL_NON_WATER, LABEL_WATER
from data.sen1floods11 import Sen1Floods11Dataset, Sen1Floods11Sample

CHIP_SIZE = 9  # matches tests/test_hand.py's valley DEM -- big enough for flow routing to settle
_TRANSFORM = from_origin(0, 0, 1, 1)

# evaluate.py's real HAND_ACCUMULATION_THRESHOLD (100) is tuned for real
# 512x512 chips; a 9x9 synthetic valley never accumulates that much flow,
# so every pixel would come back NaN. tests/test_hand.py established 15 as
# the right threshold for this exact synthetic valley shape, so tests here
# patch the constant down to that instead of changing the real one.
_SYNTHETIC_ACCUMULATION_THRESHOLD = 15


@pytest.fixture(autouse=True)
def _use_synthetic_accumulation_threshold(monkeypatch):
    # Patched on data.chip_terrain, not benchmarks.evaluate: that's where
    # compute_terrain_layers is actually defined, so that's the module
    # namespace it reads HAND_ACCUMULATION_THRESHOLD from at call time --
    # patching the name benchmarks.evaluate re-exports wouldn't affect it.
    monkeypatch.setattr(
        chip_terrain_module, "HAND_ACCUMULATION_THRESHOLD", _SYNTHETIC_ACCUMULATION_THRESHOLD
    )


def _write_s1_image(path: Path, vv_db: np.ndarray, vh_db: np.ndarray) -> None:
    data = np.stack([vv_db, vh_db], axis=0).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=2,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(data)


def _write_label(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=1,
        dtype=np.int16,
        crs="EPSG:4326",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(values.astype(np.int16), 1)


def _write_dem(path: Path, dem: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(dem.astype(np.float32), 1)


def _valley_dem() -> np.ndarray:
    """Same shape as tests/test_hand.py's make_valley_dem: a drainage line down column 2."""
    dem = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float64)
    for col in range(CHIP_SIZE):
        dem[:, col] = abs(col - 2) * 5.0 + 10.0
    for row in range(CHIP_SIZE):
        dem[row, :] += (CHIP_SIZE - row) * 0.1
    return dem


def _make_fixture(tmp_path: Path, chip_id: str, vv_db: float, vh_db: float) -> None:
    """Write one chip's S1 image, label, and DEM, all under tmp_path's real layout."""
    (tmp_path / "S1Hand").mkdir(exist_ok=True)
    (tmp_path / "LabelHand").mkdir(exist_ok=True)
    (tmp_path / "DEMHand").mkdir(exist_ok=True)

    vv = np.full((CHIP_SIZE, CHIP_SIZE), vv_db, dtype=np.float32)
    vh = np.full((CHIP_SIZE, CHIP_SIZE), vh_db, dtype=np.float32)
    _write_s1_image(tmp_path / "S1Hand" / f"{chip_id}_S1Hand.tif", vv, vh)

    label = np.full((CHIP_SIZE, CHIP_SIZE), LABEL_NON_WATER, dtype=np.int16)
    label[0, 0] = LABEL_WATER
    _write_label(tmp_path / "LabelHand" / f"{chip_id}_LabelHand.tif", label)

    _write_dem(tmp_path / "DEMHand" / f"{chip_id}_DEMHand.tif", _valley_dem())


def _write_split(tmp_path: Path, chip_ids: list[str]) -> Path:
    split_csv = tmp_path / "split.csv"
    lines = [f"{cid}_S1Hand.tif,{cid}_LabelHand.tif" for cid in chip_ids]
    split_csv.write_text("\n".join(lines) + "\n")
    return split_csv


def test_load_dem_reads_single_band(tmp_path):
    dem = _valley_dem()
    _write_dem(tmp_path / "dem.tif", dem)

    loaded = load_dem(tmp_path / "dem.tif")

    assert loaded.shape == (CHIP_SIZE, CHIP_SIZE)
    assert np.allclose(loaded, dem.astype(np.float32))


def test_compute_terrain_layers_returns_slope_and_hand_same_shape():
    dem = _valley_dem()

    slope, hand = compute_terrain_layers(dem)

    assert slope.shape == dem.shape
    assert hand.shape == dem.shape


def test_evaluate_baseline_scores_every_chip(tmp_path):
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_fixture(tmp_path, "Ghana_1", vv_db=-8.0, vh_db=-12.0)
    split_csv = _write_split(tmp_path, ["Bolivia_1", "Ghana_1"])

    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)

    results = evaluate_baseline(otsu_predict, dataset, tmp_path / "DEMHand")

    assert len(results) == 2
    assert {m.chip_id for m in results} == {"Bolivia_1", "Ghana_1"}
    assert {m.event for m in results} == {"Bolivia", "Ghana"}


def test_train_random_forest_baseline_produces_a_working_model(tmp_path):
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_fixture(tmp_path, "Ghana_1", vv_db=-8.0, vh_db=-12.0)
    split_csv = _write_split(tmp_path, ["Bolivia_1", "Ghana_1"])

    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)

    model = train_random_forest_baseline(dataset, tmp_path / "DEMHand", seed=0)

    assert hasattr(model, "predict")


def test_evaluate_baseline_rejects_missing_dem(tmp_path):
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    split_csv = _write_split(tmp_path, ["Bolivia_1"])
    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)

    with pytest.raises(rasterio.errors.RasterioIOError):
        evaluate_baseline(otsu_predict, dataset, tmp_path / "nonexistent_dem_dir")


def _all_nan_vv_sample() -> Sen1Floods11Sample:
    """A chip like the real Paraguay_34417 test chip: entirely NaN in VV."""
    image = np.full((3, CHIP_SIZE, CHIP_SIZE), np.nan, dtype=np.float32)
    label = np.full((CHIP_SIZE, CHIP_SIZE), LABEL_NON_WATER, dtype=np.int64)
    return Sen1Floods11Sample(image=image, label=label, chip_id="Paraguay_34417")


def test_otsu_predict_falls_back_to_all_non_water_on_entirely_nan_vv():
    sample = _all_nan_vv_sample()
    slope = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hand = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)

    predicted = otsu_predict(sample, slope, hand)

    assert predicted.shape == (CHIP_SIZE, CHIP_SIZE)
    assert not predicted.any()


def test_otsu_hand_predict_falls_back_to_all_non_water_on_entirely_nan_vv():
    sample = _all_nan_vv_sample()
    slope = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hand = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)

    predicted = otsu_hand_predict(sample, slope, hand)

    assert predicted.shape == (CHIP_SIZE, CHIP_SIZE)
    assert not predicted.any()


def test_build_terrain_cache_returns_one_entry_per_chip(tmp_path):
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_fixture(tmp_path, "Ghana_1", vv_db=-8.0, vh_db=-12.0)

    cache = build_terrain_cache(["Bolivia_1", "Ghana_1"], tmp_path / "DEMHand")

    assert set(cache.keys()) == {"Bolivia_1", "Ghana_1"}
    slope, hand = cache["Bolivia_1"]
    assert slope.shape == (CHIP_SIZE, CHIP_SIZE)
    assert hand.shape == (CHIP_SIZE, CHIP_SIZE)


def test_evaluate_baseline_uses_terrain_cache_instead_of_dem_dir(tmp_path):
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    split_csv = _write_split(tmp_path, ["Bolivia_1"])
    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)
    terrain_cache = build_terrain_cache(["Bolivia_1"], tmp_path / "DEMHand")

    # dem_dir points nowhere; this only succeeds if the cache is actually used.
    results = evaluate_baseline(
        otsu_predict, dataset, tmp_path / "nonexistent_dem_dir", terrain_cache
    )

    assert len(results) == 1
    assert results[0].chip_id == "Bolivia_1"


def test_train_random_forest_baseline_uses_terrain_cache_instead_of_dem_dir(tmp_path):
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_fixture(tmp_path, "Ghana_1", vv_db=-8.0, vh_db=-12.0)
    split_csv = _write_split(tmp_path, ["Bolivia_1", "Ghana_1"])
    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)
    terrain_cache = build_terrain_cache(["Bolivia_1", "Ghana_1"], tmp_path / "DEMHand")

    model = train_random_forest_baseline(
        dataset, tmp_path / "nonexistent_dem_dir", seed=0, terrain_cache=terrain_cache
    )

    assert hasattr(model, "predict")
