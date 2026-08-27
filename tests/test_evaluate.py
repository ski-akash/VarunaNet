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
    compute_per_event_otsu_thresholds,
    compute_terrain_layers,
    evaluate_baseline,
    load_dem,
    make_otsu_hand_permanent_water_predict,
    make_otsu_hand_predict,
    make_otsu_permanent_water_predict,
    make_otsu_predict,
    train_random_forest_baseline,
)
from data.chip_terrain import get_terrain
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


def _write_jrc(path: Path, values: np.ndarray) -> None:
    """Matches the real, already-binary JRCWaterHand chips (see
    benchmarks.evaluate._load_jrc_permanent_water's docstring): 1 band,
    uint8, 0/1 -- not a raw 0-100 occurrence layer."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(values.astype(np.uint8), 1)


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
    event_thresholds = compute_per_event_otsu_thresholds(dataset)

    results = evaluate_baseline(make_otsu_predict(event_thresholds), dataset, tmp_path / "DEMHand")

    assert len(results) == 2
    assert {m.chip_id for m in results} == {"Bolivia_1", "Ghana_1"}
    assert {m.event for m in results} == {"Bolivia", "Ghana"}


def test_compute_per_event_otsu_thresholds_pools_every_chip_in_the_event(tmp_path):
    # Two Bolivia chips at different constant VH values (evaluate.py
    # thresholds VH, not VV -- see its module-level _OTSU_BAND_INDEX),
    # plus a lone Ghana chip -- pooling the two Bolivia chips together
    # gives Otsu a genuine two-population histogram to split (same shape
    # as test_otsu.py's make_bimodal_band, just without noise), which a
    # single constant-valued chip alone couldn't produce.
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_fixture(tmp_path, "Bolivia_2", vv_db=-8.0, vh_db=-12.0)
    _make_fixture(tmp_path, "Ghana_1", vv_db=-14.0, vh_db=-18.0)
    split_csv = _write_split(tmp_path, ["Bolivia_1", "Bolivia_2", "Ghana_1"])
    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)

    thresholds = compute_per_event_otsu_thresholds(dataset)

    assert set(thresholds) == {"Bolivia", "Ghana"}
    # Bolivia's threshold has to land strictly between its own two chips'
    # constant VH values -- only possible if both chips' pixels actually
    # got pooled into one histogram together, not computed from just one.
    assert -25.0 < thresholds["Bolivia"] < -12.0


def test_compute_per_event_otsu_thresholds_omits_events_with_no_finite_pixels(tmp_path):
    # A real all-NaN chip on disk, like the real Paraguay_34417 test chip
    # (scene-edge no-data), not just a synthetic in-memory sample -- this
    # exercises compute_per_event_otsu_thresholds' own dataset[i] disk
    # read, not otsu_water_mask's fallback in isolation.
    (tmp_path / "S1Hand").mkdir(exist_ok=True)
    (tmp_path / "LabelHand").mkdir(exist_ok=True)
    nan_vv_vh = np.full((CHIP_SIZE, CHIP_SIZE), np.nan, dtype=np.float32)
    _write_s1_image(tmp_path / "S1Hand" / "Paraguay_1_S1Hand.tif", nan_vv_vh, nan_vv_vh)
    _write_label(
        tmp_path / "LabelHand" / "Paraguay_1_LabelHand.tif",
        np.full((CHIP_SIZE, CHIP_SIZE), LABEL_NON_WATER, dtype=np.int16),
    )
    dataset = Sen1Floods11Dataset.from_pairs(
        tmp_path / "S1Hand",
        tmp_path / "LabelHand",
        [("Paraguay_1_S1Hand.tif", "Paraguay_1_LabelHand.tif")],
    )

    thresholds = compute_per_event_otsu_thresholds(dataset)

    assert thresholds == {}


def test_otsu_permanent_water_predict_removes_flood_flags_that_are_permanent_water(tmp_path):
    # Bolivia_1 (very low VH, water-like) vs. Bolivia_2 (much higher VH,
    # land-like) gives compute_per_event_otsu_thresholds a real bimodal
    # histogram to split, same reasoning as the pooling test above -- so
    # make_otsu_predict actually flags Bolivia_1 as water somewhere before
    # permanent-water removal has anything real to remove.
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_fixture(tmp_path, "Bolivia_2", vv_db=-8.0, vh_db=-12.0)
    (tmp_path / "JRCWaterHand").mkdir(exist_ok=True)
    _write_jrc(
        tmp_path / "JRCWaterHand" / "Bolivia_1_JRCWaterHand.tif",
        np.ones((CHIP_SIZE, CHIP_SIZE)),
    )
    split_csv = _write_split(tmp_path, ["Bolivia_1", "Bolivia_2"])
    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)
    thresholds = compute_per_event_otsu_thresholds(dataset)
    sample = dataset[0]  # Bolivia_1
    slope, hand = get_terrain(sample.chip_id, tmp_path / "DEMHand", None)

    plain_prediction = make_otsu_predict(thresholds)(sample, slope, hand)
    assert plain_prediction.any(), "fixture is set up wrong if Otsu finds no water at all here"

    permanent_water_prediction = make_otsu_permanent_water_predict(
        thresholds, tmp_path / "JRCWaterHand"
    )(sample, slope, hand)
    # Bolivia_1's JRC chip is *entirely* permanent water, so every pixel
    # Otsu flagged as water must be removed -- flood extent = water minus
    # permanent water = nothing, when permanent water covers the whole chip.
    assert not permanent_water_prediction.any()


def test_otsu_hand_permanent_water_predict_removes_flood_flags_that_are_permanent_water(tmp_path):
    _make_fixture(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_fixture(tmp_path, "Bolivia_2", vv_db=-8.0, vh_db=-12.0)
    (tmp_path / "JRCWaterHand").mkdir(exist_ok=True)
    _write_jrc(
        tmp_path / "JRCWaterHand" / "Bolivia_1_JRCWaterHand.tif",
        np.ones((CHIP_SIZE, CHIP_SIZE)),
    )
    split_csv = _write_split(tmp_path, ["Bolivia_1", "Bolivia_2"])
    dataset = Sen1Floods11Dataset(tmp_path / "S1Hand", tmp_path / "LabelHand", split_csv)
    thresholds = compute_per_event_otsu_thresholds(dataset)
    sample = dataset[0]
    slope, hand = get_terrain(sample.chip_id, tmp_path / "DEMHand", None)

    prediction = make_otsu_hand_permanent_water_predict(thresholds, tmp_path / "JRCWaterHand")(
        sample, slope, hand
    )

    assert not prediction.any()


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
    event_thresholds = compute_per_event_otsu_thresholds(dataset)

    with pytest.raises(rasterio.errors.RasterioIOError):
        evaluate_baseline(
            make_otsu_predict(event_thresholds), dataset, tmp_path / "nonexistent_dem_dir"
        )


def _all_nan_vv_sample() -> Sen1Floods11Sample:
    """A chip like the real Paraguay_34417 test chip: entirely NaN in VV."""
    image = np.full((3, CHIP_SIZE, CHIP_SIZE), np.nan, dtype=np.float32)
    label = np.full((CHIP_SIZE, CHIP_SIZE), LABEL_NON_WATER, dtype=np.int64)
    return Sen1Floods11Sample(image=image, label=label, chip_id="Paraguay_34417")


def test_otsu_predict_falls_back_to_all_non_water_when_event_has_no_threshold():
    # An empty threshold map is exactly what compute_per_event_otsu_thresholds
    # would produce for an event whose every chip is entirely NaN in VV --
    # there's no finite pixel anywhere to pool a threshold from.
    sample = _all_nan_vv_sample()
    slope = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hand = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)

    predicted = make_otsu_predict({})(sample, slope, hand)

    assert predicted.shape == (CHIP_SIZE, CHIP_SIZE)
    assert not predicted.any()


def test_otsu_hand_predict_falls_back_to_all_non_water_when_event_has_no_threshold():
    sample = _all_nan_vv_sample()
    slope = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hand = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)

    predicted = make_otsu_hand_predict({})(sample, slope, hand)

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
    event_thresholds = compute_per_event_otsu_thresholds(dataset)

    # dem_dir points nowhere; this only succeeds if the cache is actually used.
    results = evaluate_baseline(
        make_otsu_predict(event_thresholds),
        dataset,
        tmp_path / "nonexistent_dem_dir",
        terrain_cache,
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
