"""
Tests for training/sen1floods11_dataset.py, using synthetic
Sen1Floods11-shaped fixture GeoTIFFs (the same style
tests/test_evaluate.py uses) -- but at the *real* data-contract chip size
(512x512, data/contract.CHIP_SIZE), unlike evaluate.py's raw-array
functions: Sen1Floods11TorchDataset validates against the actual data
contract, so a smaller fixture would fail validate_input_tensor's shape
check before ever testing anything interesting.

The synthetic DEM uses a clear monotonic gradient rather than a flat
surface deliberately -- a flat DEM is pysheds' slow path (nothing for
flow direction to resolve unambiguously), confirmed to cost noticeably
more time than a gradient DEM of the same size before choosing this shape.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from data.chip_terrain import build_terrain_cache
from data.contract import CHIP_SIZE, LABEL_NON_WATER, LABEL_WATER
from data.normalization import NormalizationStats
from training.sen1floods11_dataset import Sen1Floods11TorchDataset, build_sen1floods11_dataset

_TRANSFORM = from_origin(0, 0, 1, 1)

_FLAT_STATS = NormalizationStats(
    channels=("VV_db", "VH_db", "VV_VH_ratio", "slope", "HAND"),
    mean=(0.0, 0.0, 0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0, 1.0, 1.0),
)


def _gradient_dem() -> np.ndarray:
    dem = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float64)
    for col in range(CHIP_SIZE):
        dem[:, col] = abs(col - CHIP_SIZE // 2) * 0.05 + 10.0
    for row in range(CHIP_SIZE):
        dem[row, :] += (CHIP_SIZE - row) * 0.01
    return dem


def _write_s1_image(path: Path, vv_db: float, vh_db: float) -> None:
    data = np.stack(
        [
            np.full((CHIP_SIZE, CHIP_SIZE), vv_db, dtype=np.float32),
            np.full((CHIP_SIZE, CHIP_SIZE), vh_db, dtype=np.float32),
        ],
        axis=0,
    )
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


def _make_fixture(root: Path, chip_id: str, vv_db: float = -12.0, vh_db: float = -18.0) -> None:
    (root / "S1Hand").mkdir(exist_ok=True, parents=True)
    (root / "LabelHand").mkdir(exist_ok=True, parents=True)
    (root / "DEMHand").mkdir(exist_ok=True, parents=True)

    _write_s1_image(root / "S1Hand" / f"{chip_id}_S1Hand.tif", vv_db, vh_db)

    label = np.full((CHIP_SIZE, CHIP_SIZE), LABEL_NON_WATER, dtype=np.int16)
    label[0, 0] = LABEL_WATER
    _write_label(root / "LabelHand" / f"{chip_id}_LabelHand.tif", label)

    _write_dem(root / "DEMHand" / f"{chip_id}_DEMHand.tif", _gradient_dem())


def _write_split(root: Path, chip_ids: list[str], filename: str = "flood_train_data.csv") -> Path:
    (root / "splits").mkdir(exist_ok=True, parents=True)
    split_csv = root / "splits" / filename
    lines = [f"{cid}_S1Hand.tif,{cid}_LabelHand.tif" for cid in chip_ids]
    split_csv.write_text("\n".join(lines) + "\n")
    return split_csv


@pytest.fixture
def fixture_root(tmp_path) -> Path:
    _make_fixture(tmp_path, "Bolivia_1")
    _make_fixture(tmp_path, "Ghana_1", vv_db=-8.0, vh_db=-15.0)
    _write_split(tmp_path, ["Bolivia_1", "Ghana_1"])
    return tmp_path


def test_getitem_returns_contract_conformant_tensors(fixture_root):
    split_csv = fixture_root / "splits" / "flood_train_data.csv"
    dataset = Sen1Floods11TorchDataset(
        image_dir=fixture_root / "S1Hand",
        label_dir=fixture_root / "LabelHand",
        dem_dir=fixture_root / "DEMHand",
        split_csv=split_csv,
        normalization_stats=_FLAT_STATS,
    )

    inputs, label, _ = dataset[0]

    assert inputs.shape == (5, CHIP_SIZE, CHIP_SIZE)
    assert inputs.dtype.is_floating_point
    assert label.shape == (CHIP_SIZE, CHIP_SIZE)
    assert label.dtype == torch.int64


def test_getitem_replaces_nan_instead_of_propagating_it(fixture_root):
    # The gradient DEM's HAND channel has real NaN border pixels (flow
    # routing edge effect, see data/hand.py) -- this is the regression
    # test that they never reach the returned tensor.
    split_csv = fixture_root / "splits" / "flood_train_data.csv"
    dataset = Sen1Floods11TorchDataset(
        image_dir=fixture_root / "S1Hand",
        label_dir=fixture_root / "LabelHand",
        dem_dir=fixture_root / "DEMHand",
        split_csv=split_csv,
        normalization_stats=_FLAT_STATS,
    )

    inputs, _, _ = dataset[0]

    assert not torch.isnan(inputs).any()


def test_terrain_cache_is_used_instead_of_recomputing(fixture_root):
    # Precompute a cache with deliberately wrong (easy to spot) values for
    # one chip, then confirm the dataset returns exactly those values
    # rather than the real DEM-derived ones -- proves the cache path is
    # actually taken, not just accepted and ignored.
    fake_slope = np.full((CHIP_SIZE, CHIP_SIZE), 999.0, dtype=np.float32)
    fake_hand = np.full((CHIP_SIZE, CHIP_SIZE), 888.0, dtype=np.float32)
    terrain_cache = {"Bolivia_1": (fake_slope, fake_hand)}

    split_csv = fixture_root / "splits" / "flood_train_data.csv"
    dataset = Sen1Floods11TorchDataset(
        image_dir=fixture_root / "S1Hand",
        label_dir=fixture_root / "LabelHand",
        dem_dir=fixture_root / "DEMHand",
        split_csv=split_csv,
        normalization_stats=_FLAT_STATS,
        terrain_cache=terrain_cache,
    )

    inputs, _, _ = dataset[0]  # Bolivia_1 is index 0 in the split
    # channel 3 = slope, channel 4 = HAND (data/contract.py's CHANNELS order)
    assert inputs[3, 0, 0].item() == 999.0
    assert inputs[4, 0, 0].item() == 888.0


def test_channel_indices_slices_after_normalization_in_requested_order(fixture_root):
    # Regression test for the channel_indices ablation knob (spec section
    # 3.2 / training/conf/dataset/sen1floods11_vvvh_only.yaml and the
    # per-feature leave-one-out configs) -- nothing exercised this before,
    # even though real cluster training runs already depend on it.
    # channel_indices=[0, 1, 3, 4] drops index 2 (VV_VH_ratio), keeping
    # VV, VH, slope, HAND in that order -- confirmed here via the same
    # fake-terrain-cache trick test_terrain_cache_is_used_instead_of_recomputing
    # uses, so slope/HAND are identifiable by value at their new positions.
    fake_slope = np.full((CHIP_SIZE, CHIP_SIZE), 999.0, dtype=np.float32)
    fake_hand = np.full((CHIP_SIZE, CHIP_SIZE), 888.0, dtype=np.float32)
    terrain_cache = {"Bolivia_1": (fake_slope, fake_hand)}

    split_csv = fixture_root / "splits" / "flood_train_data.csv"
    dataset = Sen1Floods11TorchDataset(
        image_dir=fixture_root / "S1Hand",
        label_dir=fixture_root / "LabelHand",
        dem_dir=fixture_root / "DEMHand",
        split_csv=split_csv,
        normalization_stats=_FLAT_STATS,
        terrain_cache=terrain_cache,
        channel_indices=[0, 1, 3, 4],
    )

    inputs, _, _ = dataset[0]

    assert inputs.shape == (4, CHIP_SIZE, CHIP_SIZE)
    assert inputs[2, 0, 0].item() == 999.0  # slope, now at position 2
    assert inputs[3, 0, 0].item() == 888.0  # HAND, now at position 3


def test_normalization_is_actually_applied(fixture_root):
    # Bolivia_1's raw VV is a constant -12.0 dB. With non-trivial stats
    # (not the identity _FLAT_STATS), the returned tensor should reflect
    # (raw - mean) / std, not the raw value.
    stats = NormalizationStats(
        channels=("VV_db", "VH_db", "VV_VH_ratio", "slope", "HAND"),
        mean=(-12.0, -18.0, 0.0, 0.0, 0.0),
        std=(2.0, 2.0, 1.0, 1.0, 1.0),
    )
    split_csv = fixture_root / "splits" / "flood_train_data.csv"
    dataset = Sen1Floods11TorchDataset(
        image_dir=fixture_root / "S1Hand",
        label_dir=fixture_root / "LabelHand",
        dem_dir=fixture_root / "DEMHand",
        split_csv=split_csv,
        normalization_stats=stats,
    )

    inputs, _, _ = dataset[0]
    # (-12.0 - (-12.0)) / 2.0 == 0.0
    assert abs(inputs[0, 5, 5].item()) < 1e-4


def test_build_sen1floods11_dataset_matches_real_layout(fixture_root):
    dataset = build_sen1floods11_dataset(
        data_root=fixture_root,
        split_csv_name="flood_train_data.csv",
        normalization_stats=_FLAT_STATS,
        precompute_terrain=False,
    )

    assert len(dataset) == 2
    inputs, label, _ = dataset[0]
    assert inputs.shape == (5, CHIP_SIZE, CHIP_SIZE)


def test_build_sen1floods11_dataset_precomputes_terrain_cache(fixture_root):
    dataset = build_sen1floods11_dataset(
        data_root=fixture_root,
        split_csv_name="flood_train_data.csv",
        normalization_stats=_FLAT_STATS,
        precompute_terrain=True,
    )

    assert dataset._terrain_cache is not None
    assert set(dataset._terrain_cache) == {"Bolivia_1", "Ghana_1"}


def test_dataset_output_matches_directly_built_terrain_cache(fixture_root):
    # End-to-end sanity check: what the dataset returns for a chip should
    # be identical whether terrain came from an explicit cache built via
    # data.chip_terrain.build_terrain_cache, or was computed inline --
    # confirms the two code paths (training/sen1floods11_dataset.py and
    # benchmarks/evaluate.py) that both consume data.chip_terrain agree.
    terrain_cache = build_terrain_cache(["Bolivia_1"], fixture_root / "DEMHand")

    split_csv = fixture_root / "splits" / "flood_train_data.csv"
    with_cache = Sen1Floods11TorchDataset(
        image_dir=fixture_root / "S1Hand",
        label_dir=fixture_root / "LabelHand",
        dem_dir=fixture_root / "DEMHand",
        split_csv=split_csv,
        normalization_stats=_FLAT_STATS,
        terrain_cache=terrain_cache,
    )
    without_cache = Sen1Floods11TorchDataset(
        image_dir=fixture_root / "S1Hand",
        label_dir=fixture_root / "LabelHand",
        dem_dir=fixture_root / "DEMHand",
        split_csv=split_csv,
        normalization_stats=_FLAT_STATS,
        terrain_cache=None,
    )

    inputs_cached, _, _ = with_cache[0]
    inputs_direct, _, _ = without_cache[0]
    assert torch.equal(inputs_cached, inputs_direct)
