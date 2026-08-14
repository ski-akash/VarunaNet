"""
Tests for hold-one-event-out cross-validation, using tiny synthetic
Sen1Floods11-shaped fixtures spread across three "official-style" split
files, the same way tests/test_evaluate.py does for a single split.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import benchmarks.evaluate as evaluate_module
from benchmarks.hold_one_event_out import (
    group_pairs_by_event,
    load_all_pairs,
    run_hold_one_event_out,
)
from data.contract import LABEL_NON_WATER, LABEL_WATER

CHIP_SIZE = 9  # matches tests/test_evaluate.py -- big enough for flow routing to settle
_TRANSFORM = from_origin(0, 0, 1, 1)

# See tests/test_evaluate.py: the real HAND_ACCUMULATION_THRESHOLD (100) is
# tuned for real 512x512 chips and never triggers on a 9x9 synthetic valley.
_SYNTHETIC_ACCUMULATION_THRESHOLD = 15


@pytest.fixture(autouse=True)
def _use_synthetic_accumulation_threshold(monkeypatch):
    monkeypatch.setattr(
        evaluate_module, "HAND_ACCUMULATION_THRESHOLD", _SYNTHETIC_ACCUMULATION_THRESHOLD
    )


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


def _write_label(path: Path) -> None:
    values = np.full((CHIP_SIZE, CHIP_SIZE), LABEL_NON_WATER, dtype=np.int16)
    values[0, 0] = LABEL_WATER
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
        dst.write(values, 1)


def _valley_dem() -> np.ndarray:
    """Same shape as tests/test_hand.py's make_valley_dem: a drainage line down column 2."""
    dem = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float64)
    for col in range(CHIP_SIZE):
        dem[:, col] = abs(col - 2) * 5.0 + 10.0
    for row in range(CHIP_SIZE):
        dem[row, :] += (CHIP_SIZE - row) * 0.1
    return dem


def _write_dem(path: Path) -> None:
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
        dst.write(_valley_dem().astype(np.float32), 1)


def _make_chip(tmp_path: Path, chip_id: str, vv_db: float, vh_db: float) -> None:
    (tmp_path / "S1Hand").mkdir(exist_ok=True)
    (tmp_path / "LabelHand").mkdir(exist_ok=True)
    (tmp_path / "DEMHand").mkdir(exist_ok=True)
    _write_s1_image(tmp_path / "S1Hand" / f"{chip_id}_S1Hand.tif", vv_db, vh_db)
    _write_label(tmp_path / "LabelHand" / f"{chip_id}_LabelHand.tif")
    _write_dem(tmp_path / "DEMHand" / f"{chip_id}_DEMHand.tif")


def _write_splits(tmp_path: Path, train: list[str], valid: list[str], test: list[str]) -> Path:
    """Write the three official-style split CSVs, distributing the given chip ids across them."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(exist_ok=True)
    for filename, chip_ids in (
        ("flood_train_data.csv", train),
        ("flood_valid_data.csv", valid),
        ("flood_test_data.csv", test),
    ):
        lines = [f"{cid}_S1Hand.tif,{cid}_LabelHand.tif" for cid in chip_ids]
        (splits_dir / filename).write_text("\n".join(lines) + "\n" if lines else "")
    return splits_dir


def test_load_all_pairs_pools_three_splits(tmp_path):
    splits_dir = _write_splits(tmp_path, train=["Alpha_1"], valid=["Beta_1"], test=["Gamma_1"])

    pairs = load_all_pairs(splits_dir)

    assert pairs == [
        ("Alpha_1_S1Hand.tif", "Alpha_1_LabelHand.tif"),
        ("Beta_1_S1Hand.tif", "Beta_1_LabelHand.tif"),
        ("Gamma_1_S1Hand.tif", "Gamma_1_LabelHand.tif"),
    ]


def test_group_pairs_by_event_groups_correctly():
    pairs = [
        ("Alpha_1_S1Hand.tif", "Alpha_1_LabelHand.tif"),
        ("Alpha_2_S1Hand.tif", "Alpha_2_LabelHand.tif"),
        ("Beta_1_S1Hand.tif", "Beta_1_LabelHand.tif"),
    ]

    groups = group_pairs_by_event(pairs)

    assert set(groups.keys()) == {"Alpha", "Beta"}
    assert len(groups["Alpha"]) == 2
    assert len(groups["Beta"]) == 1


def test_run_hold_one_event_out_scores_every_chip_exactly_once(tmp_path):
    # Three events, two chips each, deliberately spread across all three
    # official-style splits so pooling+regrouping is actually exercised.
    _make_chip(tmp_path, "Alpha_1", vv_db=-20.0, vh_db=-25.0)
    _make_chip(tmp_path, "Alpha_2", vv_db=-8.0, vh_db=-12.0)
    _make_chip(tmp_path, "Beta_1", vv_db=-20.0, vh_db=-25.0)
    _make_chip(tmp_path, "Beta_2", vv_db=-8.0, vh_db=-12.0)
    _make_chip(tmp_path, "Gamma_1", vv_db=-20.0, vh_db=-25.0)
    _make_chip(tmp_path, "Gamma_2", vv_db=-8.0, vh_db=-12.0)
    splits_dir = _write_splits(
        tmp_path,
        train=["Alpha_1", "Beta_1", "Gamma_1"],
        valid=["Alpha_2"],
        test=["Beta_2", "Gamma_2"],
    )

    results = run_hold_one_event_out(
        image_dir=tmp_path / "S1Hand",
        label_dir=tmp_path / "LabelHand",
        dem_dir=tmp_path / "DEMHand",
        splits_dir=splits_dir,
        seed=0,
    )

    assert set(results.keys()) == {"Otsu", "Otsu + HAND", "Random Forest"}
    all_chip_ids = {"Alpha_1", "Alpha_2", "Beta_1", "Beta_2", "Gamma_1", "Gamma_2"}
    for name, chip_metrics in results.items():
        # Every chip scored exactly once, regardless of which official
        # split it originally came from.
        assert len(chip_metrics) == 6, name
        assert {m.chip_id for m in chip_metrics} == all_chip_ids, name
