"""Tests for the metrics module, using small synthetic masks and labels."""

import numpy as np
import pytest

from benchmarks.metrics import (
    ChipMetrics,
    ConfusionCounts,
    compute_chip_metrics,
    confusion_counts,
    event_name,
    f1_score,
    iou_score,
    precision_score,
    recall_score,
    summarize,
    summarize_per_event,
)
from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER


def test_confusion_counts_basic():
    # 2x2 chip: one true positive, one false positive, one false negative,
    # one true negative.
    predicted = np.array([[True, True], [False, False]])
    label = np.array([[LABEL_WATER, LABEL_NON_WATER], [LABEL_WATER, LABEL_NON_WATER]])

    counts = confusion_counts(predicted, label)

    assert counts == ConfusionCounts(
        true_positive=1, false_positive=1, false_negative=1, true_negative=1
    )


def test_confusion_counts_excludes_ignore_pixels():
    predicted = np.array([[True, True]])
    label = np.array([[LABEL_WATER, LABEL_IGNORE]])

    counts = confusion_counts(predicted, label)

    # The ignore pixel contributes to neither true nor false positives.
    assert counts == ConfusionCounts(
        true_positive=1, false_positive=0, false_negative=0, true_negative=0
    )


def test_confusion_counts_rejects_shape_mismatch():
    predicted = np.zeros((2, 2), dtype=bool)
    label = np.zeros((2, 3), dtype=np.int64)

    with pytest.raises(ValueError, match="shape mismatch"):
        confusion_counts(predicted, label)


def test_iou_precision_recall_f1_perfect_prediction():
    counts = ConfusionCounts(true_positive=10, false_positive=0, false_negative=0, true_negative=90)

    assert iou_score(counts) == 1.0
    assert precision_score(counts) == 1.0
    assert recall_score(counts) == 1.0
    assert f1_score(counts) == 1.0


def test_iou_precision_recall_f1_known_values():
    # 5 correctly predicted, 3 false positives, 2 false negatives.
    counts = ConfusionCounts(true_positive=5, false_positive=3, false_negative=2, true_negative=90)

    assert iou_score(counts) == pytest.approx(5 / 10)
    assert precision_score(counts) == pytest.approx(5 / 8)
    assert recall_score(counts) == pytest.approx(5 / 7)
    assert f1_score(counts) == pytest.approx(2 * (5 / 8) * (5 / 7) / ((5 / 8) + (5 / 7)))


def test_metrics_are_nan_when_denominator_is_zero():
    # No water anywhere -- ground truth and prediction both entirely dry.
    counts = ConfusionCounts(true_positive=0, false_positive=0, false_negative=0, true_negative=100)

    assert np.isnan(iou_score(counts))
    assert np.isnan(precision_score(counts))
    assert np.isnan(recall_score(counts))
    assert np.isnan(f1_score(counts))


def test_event_name_parses_chip_id():
    assert event_name("Bolivia_103757") == "Bolivia"
    assert event_name("Sri-Lanka_57389") == "Sri-Lanka"


def test_compute_chip_metrics_end_to_end():
    predicted = np.array([[True, False], [False, False]])
    label = np.array([[LABEL_WATER, LABEL_NON_WATER], [LABEL_NON_WATER, LABEL_NON_WATER]])

    metrics = compute_chip_metrics("Ghana_1033830", predicted, label)

    assert metrics.chip_id == "Ghana_1033830"
    assert metrics.event == "Ghana"
    assert metrics.iou == 1.0
    assert metrics.f1 == 1.0


def test_summarize_reports_mean_and_median():
    metrics = [
        ChipMetrics(chip_id="a", event="X", iou=0.2, f1=0.3, precision=0.4, recall=0.5),
        ChipMetrics(chip_id="b", event="X", iou=0.4, f1=0.5, precision=0.6, recall=0.7),
        ChipMetrics(chip_id="c", event="X", iou=0.9, f1=0.9, precision=0.9, recall=0.9),
    ]

    summary = summarize(metrics)

    assert summary.n_chips == 3
    assert summary.mean_iou == pytest.approx((0.2 + 0.4 + 0.9) / 3)
    assert summary.median_iou == pytest.approx(0.4)


def test_summarize_skips_nan_chips_in_mean_and_median():
    nan = float("nan")
    metrics = [
        ChipMetrics(chip_id="a", event="X", iou=0.5, f1=0.5, precision=0.5, recall=0.5),
        ChipMetrics(chip_id="b", event="X", iou=nan, f1=nan, precision=nan, recall=nan),
    ]

    summary = summarize(metrics)

    # The NaN chip is excluded, not treated as 0 -- otherwise a
    # correctly-predicted dry chip would drag the average down.
    assert summary.mean_iou == pytest.approx(0.5)
    assert summary.n_chips == 2  # n_chips still counts every chip passed in


def test_summarize_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_summarize_per_event_groups_correctly():
    metrics = [
        ChipMetrics(
            chip_id="Bolivia_1", event="Bolivia", iou=0.2, f1=0.2, precision=0.2, recall=0.2
        ),
        ChipMetrics(
            chip_id="Bolivia_2", event="Bolivia", iou=0.6, f1=0.6, precision=0.6, recall=0.6
        ),
        ChipMetrics(chip_id="Ghana_1", event="Ghana", iou=0.9, f1=0.9, precision=0.9, recall=0.9),
    ]

    per_event = summarize_per_event(metrics)

    assert set(per_event.keys()) == {"Bolivia", "Ghana"}
    assert per_event["Bolivia"].n_chips == 2
    assert per_event["Bolivia"].mean_iou == pytest.approx(0.4)
    assert per_event["Ghana"].n_chips == 1
    assert per_event["Ghana"].mean_iou == pytest.approx(0.9)
