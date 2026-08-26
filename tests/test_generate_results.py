"""
Tests for benchmarks/generate_results.py -- the report formatting/
interpretation helpers get direct unit tests on synthetic summaries;
run_official_split gets an integration-style test against tiny synthetic
chips, the same fixture style as tests/test_evaluate.py.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import data.chip_terrain as chip_terrain_module
from benchmarks.generate_results import (
    _best_baseline,
    _describe_extreme_event,
    _hardest_and_easiest_events,
    _per_event_table,
    _summary_table,
    render_report,
    run_official_split,
)
from benchmarks.metrics import AggregateMetrics, MetricSummary
from data.contract import LABEL_NON_WATER, LABEL_WATER

CHIP_SIZE = 9
_TRANSFORM = from_origin(0, 0, 1, 1)
_SYNTHETIC_ACCUMULATION_THRESHOLD = 15  # see tests/test_evaluate.py for why


@pytest.fixture(autouse=True)
def _use_synthetic_accumulation_threshold(monkeypatch):
    # Patched on data.chip_terrain -- see tests/test_evaluate.py's version
    # of this fixture for why it can't be patched on benchmarks.evaluate.
    monkeypatch.setattr(
        chip_terrain_module, "HAND_ACCUMULATION_THRESHOLD", _SYNTHETIC_ACCUMULATION_THRESHOLD
    )


def _summary(mean_iou: float, median_iou: float = 0.0, n_chips: int = 1) -> MetricSummary:
    return MetricSummary(
        mean_iou=mean_iou,
        median_iou=median_iou,
        mean_f1=mean_iou,
        mean_precision=mean_iou,
        mean_recall=mean_iou,
        n_chips=n_chips,
    )


def _pooled(iou: float, n_chips: int = 1) -> AggregateMetrics:
    return AggregateMetrics(
        iou=iou,
        f1=iou,
        precision=iou,
        recall=iou,
        overall_accuracy=iou,
        kappa=iou,
        n_chips=n_chips,
        n_valid_pixels=n_chips * 100,
    )


def test_summary_table_has_a_row_per_baseline():
    table = _summary_table(
        {"Otsu": _summary(0.2), "Random Forest": _summary(0.3)},
        {"Otsu": _pooled(0.5), "Random Forest": _pooled(0.6)},
    )

    assert "| Otsu |" in table
    assert "| Random Forest |" in table
    assert "0.200" in table
    assert "0.300" in table


def test_summary_table_reports_pooled_iou_alongside_the_per_chip_mean():
    """
    Guards the regression this column exists to fix: RESULTS.md previously
    carried per-chip means only, so the project's headline metric was not
    written down anywhere a reader could find it.
    """
    table = _summary_table({"Otsu": _summary(0.304)}, {"Otsu": _pooled(0.479)})

    assert "Pooled IoU" in table
    assert "0.479" in table  # pooled
    assert "0.304" in table  # per-chip, still present


def test_per_event_table_marks_missing_events_with_a_dash():
    per_baseline_events = {
        "Otsu": {"Ghana": _summary(0.1, n_chips=5), "India": _summary(0.4, n_chips=3)},
        "Random Forest": {"Ghana": _summary(0.2, n_chips=5)},  # missing India
    }

    table = _per_event_table(per_baseline_events)

    lines = table.splitlines()
    india_row = next(line for line in lines if line.startswith("| India"))
    assert "0.400" in india_row
    assert "-" in india_row  # Random Forest has no India entry


def test_best_baseline_picks_highest_pooled_iou():
    # Deliberately ordered so the per-chip mean would pick a different
    # winner than pooled IoU does -- the ranking must follow the headline
    # (pooled) metric, not the per-chip one.
    pooled = {
        "Otsu": _pooled(0.60),
        "Random Forest": _pooled(0.40),
        "Otsu + HAND": _pooled(0.65),
    }

    assert _best_baseline(pooled) == "Otsu + HAND"


def test_hardest_and_easiest_events_by_mean_iou():
    per_event = {"Ghana": _summary(0.05), "Spain": _summary(0.4), "USA": _summary(0.15)}

    hardest, easiest = _hardest_and_easiest_events(per_event)

    assert hardest == "Ghana"
    assert easiest == "Spain"


def test_describe_extreme_event_reports_every_baseline_when_they_agree():
    per_baseline_events = {
        "Otsu": {"Ghana": _summary(0.05), "Spain": _summary(0.4)},
        "Random Forest": {"Ghana": _summary(0.1), "Spain": _summary(0.5)},
    }

    # Every baseline's own hardest event is Ghana -- should not be hedged.
    assert _describe_extreme_event("Ghana", per_baseline_events, pick_hardest=True) == (
        "every baseline"
    )


def test_describe_extreme_event_names_the_disagreeing_baseline():
    # Otsu's own hardest event is Ghana (0.05); Random Forest's own hardest
    # event is Somalia (0.1) -- Ghana is NOT Random Forest's hardest, so
    # claiming "every baseline" for Ghana would be wrong (this is the exact
    # bug found in a real generated report: Somalia was reported as the
    # hardest event "for every baseline" when only Random Forest agreed).
    per_baseline_events = {
        "Otsu": {"Ghana": _summary(0.05), "Somalia": _summary(0.2), "Spain": _summary(0.4)},
        "Random Forest": {"Ghana": _summary(0.3), "Somalia": _summary(0.1), "Spain": _summary(0.5)},
    }

    description = _describe_extreme_event("Ghana", per_baseline_events, pick_hardest=True)

    assert description == "Otsu only -- other baselines disagree"


def test_render_report_includes_both_sections_and_baseline_names():
    summaries = {
        "Otsu": _summary(0.2, n_chips=10),
        "Otsu + HAND": _summary(0.19, n_chips=10),
        "Random Forest": _summary(0.25, n_chips=10),
    }
    per_event = {name: {"Ghana": _summary(0.2, n_chips=10)} for name in summaries}

    pooled = {name: _pooled(0.5, n_chips=10) for name in summaries}

    report = render_report(summaries, per_event, summaries, per_event, pooled, pooled)

    assert "# Benchmark Results" in report
    assert "Pooled IoU" in report
    assert "Official train/val/test split" in report
    assert "Hold-one-event-out cross-validation" in report
    assert "Random Forest" in report
    assert "Ghana" in report


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


def test_run_official_split_scores_only_test_chips(tmp_path):
    _make_chip(tmp_path, "Bolivia_1", vv_db=-20.0, vh_db=-25.0)
    _make_chip(tmp_path, "Bolivia_2", vv_db=-8.0, vh_db=-12.0)
    _make_chip(tmp_path, "Ghana_1", vv_db=-20.0, vh_db=-25.0)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "flood_train_data.csv").write_text(
        "Bolivia_1_S1Hand.tif,Bolivia_1_LabelHand.tif\n"
        "Bolivia_2_S1Hand.tif,Bolivia_2_LabelHand.tif\n"
    )
    (splits_dir / "flood_valid_data.csv").write_text("")
    (splits_dir / "flood_test_data.csv").write_text("Ghana_1_S1Hand.tif,Ghana_1_LabelHand.tif\n")

    results = run_official_split(
        tmp_path / "S1Hand", tmp_path / "LabelHand", tmp_path / "DEMHand", splits_dir
    )

    assert set(results.keys()) == {"Otsu", "Otsu + HAND", "Random Forest"}
    for name, chip_metrics in results.items():
        assert [m.chip_id for m in chip_metrics] == ["Ghana_1"], name
