"""Tests for the metrics module, using small synthetic masks and labels."""

import numpy as np
import pytest

from benchmarks.metrics import (
    ChipMetrics,
    ConfusionCounts,
    aggregate_metrics,
    cohens_kappa,
    compute_chip_metrics,
    confusion_counts,
    event_name,
    f1_score,
    iou_score,
    overall_accuracy,
    pool_counts,
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


# --- Overall Accuracy, Kappa, and pooled/aggregate metrics --------------------


def test_overall_accuracy_basic():
    # 6 correct out of 10 valid pixels.
    counts = ConfusionCounts(true_positive=2, true_negative=4, false_positive=3, false_negative=1)
    assert overall_accuracy(counts) == pytest.approx(0.6)


def test_overall_accuracy_undefined_with_no_valid_pixels():
    counts = ConfusionCounts(true_positive=0, true_negative=0, false_positive=0, false_negative=0)
    assert np.isnan(overall_accuracy(counts))


def test_cohens_kappa_perfect_agreement_is_one():
    counts = ConfusionCounts(true_positive=30, true_negative=70, false_positive=0, false_negative=0)
    assert cohens_kappa(counts) == pytest.approx(1.0)


def test_cohens_kappa_matches_hand_computed_value():
    # po = (20+60)/100 = 0.80
    # pe = ((20+10)*(20+10) + (10+60)*(10+60)) / 100^2 = (900 + 4900)/10000 = 0.58
    # kappa = (0.80 - 0.58) / (1 - 0.58) = 0.22/0.42
    counts = ConfusionCounts(
        true_positive=20, true_negative=60, false_positive=10, false_negative=10
    )
    assert cohens_kappa(counts) == pytest.approx(0.22 / 0.42)


def test_cohens_kappa_undefined_when_expected_agreement_is_total():
    # Everything is non-water and everything was predicted non-water: observed
    # agreement is perfect but chance agreement is also 1, so kappa is 0/0.
    counts = ConfusionCounts(true_positive=0, true_negative=50, false_positive=0, false_negative=0)
    assert np.isnan(cohens_kappa(counts))


def test_all_dry_prediction_scores_high_overall_accuracy_but_zero_iou():
    """
    The single most important property in this module: a model that detects
    nothing still posts a high Overall Accuracy on an imbalanced flood scene,
    while IoU correctly reports it as useless. This is the evidence behind
    reporting IoU as primary and treating the published literature's OA
    figures as not directly meaningful (spec section 15.0).
    """
    # 5% of pixels are water -- a realistic flood-chip imbalance.
    label = np.full((10, 10), LABEL_NON_WATER)
    label[0, :5] = LABEL_WATER
    predicted_nothing = np.zeros((10, 10), dtype=bool)

    counts = confusion_counts(predicted_nothing, label)

    assert overall_accuracy(counts) == pytest.approx(0.95)
    assert iou_score(counts) == pytest.approx(0.0)
    # Kappa correctly refuses to reward the constant predictor the way OA does.
    assert cohens_kappa(counts) == pytest.approx(0.0)


def test_pool_counts_sums_across_chips():
    metrics = [
        compute_chip_metrics(
            "Ghana_1",
            np.array([[True, False]]),
            np.array([[LABEL_WATER, LABEL_NON_WATER]]),
        ),
        compute_chip_metrics(
            "Ghana_2",
            np.array([[True, True]]),
            np.array([[LABEL_WATER, LABEL_NON_WATER]]),
        ),
    ]

    pooled = pool_counts(metrics)

    assert pooled == ConfusionCounts(
        true_positive=2, true_negative=1, false_positive=1, false_negative=0
    )


def test_pool_counts_rejects_chip_metrics_without_counts():
    # Hand-constructed ChipMetrics carry no confusion counts; silently skipping
    # them would change the denominator and yield a wrong-but-plausible number.
    hand_made = [ChipMetrics(chip_id="a", event="X", iou=0.5, f1=0.5, precision=0.5, recall=0.5)]

    with pytest.raises(ValueError, match="no confusion counts"):
        pool_counts(hand_made)


def test_aggregate_iou_differs_from_per_chip_mean_iou():
    """
    Pooled IoU and per-chip-mean IoU are genuinely different numbers, and this
    test pins down the direction of the difference that motivated adding the
    pooled version at all: a tiny chip scoring 0 drags the per-chip mean down
    far more than it moves the pixel-weighted pooled value.
    """
    # Chip A: large and perfectly predicted (100 water px, all correct).
    label_a = np.full((10, 20), LABEL_NON_WATER)
    label_a[:, :10] = LABEL_WATER
    predicted_a = label_a == LABEL_WATER

    # Chip B: tiny amount of water, entirely missed -> IoU 0.
    label_b = np.full((10, 20), LABEL_NON_WATER)
    label_b[0, 0] = LABEL_WATER
    predicted_b = np.zeros((10, 20), dtype=bool)

    metrics = [
        compute_chip_metrics("Ghana_1", predicted_a, label_a),
        compute_chip_metrics("Ghana_2", predicted_b, label_b),
    ]

    per_chip = summarize(metrics)
    pooled = aggregate_metrics(metrics)

    # Per-chip mean averages 1.0 and 0.0 with equal weight.
    assert per_chip.mean_iou == pytest.approx(0.5)
    # Pooled weights by pixel: 100 correct water px vs 1 missed -> ~0.99.
    assert pooled.iou == pytest.approx(100 / 101)
    assert pooled.iou > per_chip.mean_iou


def test_aggregate_metrics_reports_chip_and_pixel_counts():
    metrics = [
        compute_chip_metrics(
            "Ghana_1",
            np.array([[True, False]]),
            np.array([[LABEL_WATER, LABEL_IGNORE]]),
        ),
        compute_chip_metrics(
            "Ghana_2",
            np.array([[True, True]]),
            np.array([[LABEL_WATER, LABEL_NON_WATER]]),
        ),
    ]

    pooled = aggregate_metrics(metrics)

    assert pooled.n_chips == 2
    # Ignore pixels are excluded from the valid-pixel total, not counted as
    # correct or incorrect: 1 valid px from chip 1 + 2 from chip 2.
    assert pooled.n_valid_pixels == 3


def test_aggregate_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="empty list"):
        aggregate_metrics([])
