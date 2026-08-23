"""
The metrics module: turns a predicted water mask and a ground-truth label
into the numbers the whole benchmark table is built from.

The spec is explicit that aggregate accuracy alone is a documented failure
mode for this kind of project -- a model can look good on average while
silently failing on one flood event's terrain or backscatter distribution.
So this module is built around two ideas kept separate on purpose:
  1. per-chip metrics (IoU, F1, precision, recall), computed once per chip
  2. summaries over a set of chips, computable either across everything
     (the aggregate) or grouped by flood event (the per-event breakdown)
so that "does this look good on average" and "does this look good on
every event" are always two different numbers, never one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from data.contract import LABEL_IGNORE, LABEL_WATER


@dataclass
class ConfusionCounts:
    """Pixel counts for the water class, over valid (non-ignore) pixels only."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


def confusion_counts(predicted_water: np.ndarray, label: np.ndarray) -> ConfusionCounts:
    """
    Count true/false positives/negatives for the water class.

    `predicted_water` is a boolean [H, W] mask; `label` is the [H, W] int
    label array from the data contract. Pixels marked LABEL_IGNORE (no-data)
    are excluded entirely -- they're not "wrong" or "right", they're outside
    the region the ground truth even covers, so they must not be counted
    either way.
    """
    if predicted_water.shape != label.shape:
        raise ValueError(f"shape mismatch: predicted {predicted_water.shape}, label {label.shape}")

    valid = label != LABEL_IGNORE
    predicted = predicted_water[valid].astype(bool)
    actual = label[valid] == LABEL_WATER

    return ConfusionCounts(
        true_positive=int(np.sum(predicted & actual)),
        false_positive=int(np.sum(predicted & ~actual)),
        false_negative=int(np.sum(~predicted & actual)),
        true_negative=int(np.sum(~predicted & ~actual)),
    )


# A chip can legitimately have zero water pixels in the ground truth (a dry
# chip) or zero predicted water pixels (a model that predicted nothing).
# When the denominator these metrics depend on is zero, the metric is
# genuinely undefined, not zero -- scoring it 0.0 would incorrectly punish
# a model for correctly predicting "no water" on a dry chip. NaN is used
# instead, and summarize() below uses NaN-aware mean/median so undefined
# chips are skipped rather than dragging the average down.
def iou_score(counts: ConfusionCounts) -> float:
    """Intersection-over-Union for the water class: TP / (TP + FP + FN)."""
    union = counts.true_positive + counts.false_positive + counts.false_negative
    return counts.true_positive / union if union > 0 else float("nan")


def precision_score(counts: ConfusionCounts) -> float:
    """TP / (TP + FP): of the pixels predicted water, how many actually were."""
    predicted_positive = counts.true_positive + counts.false_positive
    return counts.true_positive / predicted_positive if predicted_positive > 0 else float("nan")


def recall_score(counts: ConfusionCounts) -> float:
    """TP / (TP + FN): of the pixels that actually were water, how many were caught."""
    actual_positive = counts.true_positive + counts.false_negative
    return counts.true_positive / actual_positive if actual_positive > 0 else float("nan")


def f1_score(counts: ConfusionCounts) -> float:
    """Harmonic mean of precision and recall."""
    precision = precision_score(counts)
    recall = recall_score(counts)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


# --- Metrics the published Assam/Brahmaputra literature reports ---------------
#
# The SAR flood-mapping papers for this project's actual target region
# (see spec section 15.0) do not report IoU at all -- they report Overall
# Accuracy and Cohen's Kappa (e.g. OA 93.6%/95.15% for the Brahmaputra
# change-detection study, OA >82% for the RF/SVM/CART hazard-mapping study).
# To compare against them at all, this project has to speak their metric.
#
# Both are computed from the same ConfusionCounts the IoU metrics above use,
# so they cost one extra arithmetic step, not a second pass over the data.
#
# IMPORTANT, and the reason these live here with a warning attached: Overall
# Accuracy is a *bad* metric for flood segmentation. Flood water is a small
# minority of pixels, so "predict dry everywhere" scores very high OA while
# being useless. That is not a rhetorical point -- benchmarks/evaluate.py
# includes an explicit all-dry baseline so the size of that free score can be
# measured rather than asserted. Report OA to be comparable, report IoU to be
# correct, and always show both together.
def overall_accuracy(counts: ConfusionCounts) -> float:
    """(TP + TN) / all valid pixels: the fraction of pixels classified correctly."""
    total = (
        counts.true_positive + counts.true_negative + counts.false_positive + counts.false_negative
    )
    return (counts.true_positive + counts.true_negative) / total if total > 0 else float("nan")


def cohens_kappa(counts: ConfusionCounts) -> float:
    """
    Cohen's Kappa: agreement corrected for the agreement you'd expect by
    chance alone. This is why the literature reports it alongside OA -- it
    partially deflates the "predict dry everywhere" score that makes raw OA
    misleading on an imbalanced problem, because a constant predictor has
    close to zero Kappa no matter how high its OA is.

    kappa = (po - pe) / (1 - pe), where po is observed agreement (= OA) and
    pe is the agreement expected from the two marginals alone.
    """
    tp, tn = counts.true_positive, counts.true_negative
    fp, fn = counts.false_positive, counts.false_negative
    total = tp + tn + fp + fn
    if total == 0:
        return float("nan")

    observed = (tp + tn) / total
    # Expected agreement: for each class, P(predicted that class) * P(actually
    # that class), summed. Computed in one expression over the marginals to
    # avoid four intermediate divisions rounding differently.
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)

    # pe == 1 means both marginals are entirely one class (e.g. a chip that is
    # all non-water, predicted all non-water). Agreement is perfect but
    # chance-corrected agreement is genuinely undefined -- 0/0 -- so NaN, the
    # same convention the IoU metrics above use for undefined-not-zero.
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


@dataclass
class ChipMetrics:
    chip_id: str
    event: str
    iou: float
    f1: float
    precision: float
    recall: float
    # The raw confusion counts this chip's metrics were derived from, kept so
    # aggregate_metrics() below can pool them across chips without a second
    # pass over the imagery. Optional (defaults to None) only because tests
    # and older callers construct ChipMetrics by hand with literal scores;
    # compute_chip_metrics() always populates it, and aggregate_metrics()
    # raises a clear error rather than silently skipping chips that lack it.
    counts: ConfusionCounts | None = None


def event_name(chip_id: str) -> str:
    """
    Extract the flood-event name from a Sen1Floods11 chip id, e.g.
    "Bolivia_103757" -> "Bolivia", "Sri-Lanka_57389" -> "Sri-Lanka".
    Event names themselves can contain underscores in principle, but in
    Sen1Floods11 they don't (multi-word events use a hyphen, e.g.
    "Sri-Lanka"), so splitting off the last underscore-separated field
    -- the numeric chip id -- reliably recovers the event name.
    """
    return chip_id.rsplit("_", 1)[0]


def compute_chip_metrics(
    chip_id: str, predicted_water: np.ndarray, label: np.ndarray
) -> ChipMetrics:
    """Compute all four metrics for one chip in a single confusion-count pass."""
    counts = confusion_counts(predicted_water, label)
    return ChipMetrics(
        chip_id=chip_id,
        event=event_name(chip_id),
        iou=iou_score(counts),
        f1=f1_score(counts),
        precision=precision_score(counts),
        recall=recall_score(counts),
        counts=counts,
    )


@dataclass
class MetricSummary:
    mean_iou: float
    median_iou: float
    mean_f1: float
    mean_precision: float
    mean_recall: float
    n_chips: int


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _nanmedian(values: list[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(statistics.median(finite)) if finite else float("nan")


def summarize(chip_metrics: list[ChipMetrics]) -> MetricSummary:
    """
    Aggregate per-chip metrics into one summary. Mean AND median IoU are
    both reported deliberately: a mean can be dragged up by a handful of
    easy chips while most chips do poorly, and median is a cheap way to
    catch that the way the per-event breakdown catches per-event failures.
    """
    if not chip_metrics:
        raise ValueError("cannot summarize an empty list of chip metrics")

    return MetricSummary(
        mean_iou=_nanmean([m.iou for m in chip_metrics]),
        median_iou=_nanmedian([m.iou for m in chip_metrics]),
        mean_f1=_nanmean([m.f1 for m in chip_metrics]),
        mean_precision=_nanmean([m.precision for m in chip_metrics]),
        mean_recall=_nanmean([m.recall for m in chip_metrics]),
        n_chips=len(chip_metrics),
    )


# --- Aggregate (pooled) metrics ----------------------------------------------
#
# summarize() above computes a metric per chip and then averages those numbers.
# aggregate_metrics() below does something genuinely different: it pools the
# raw TP/FP/FN/TN counts across every chip *first*, then computes one metric
# from the combined counts.
#
# These are not the same number and the difference is not small. Per-chip-mean
# IoU gives every chip equal weight regardless of how much water is in it, so a
# chip with 50 water pixels counts as much as one with 50,000 -- and chips with
# tiny water fractions score near zero and drag the mean down hard. Pooled IoU
# weights by pixel, which is what the segmentation literature (including
# Sen1Floods11's own published numbers, and the ~0.72 IoU SOTA figures) reports.
#
# Both are kept, deliberately. Pooled is the number to quote when comparing
# against published work. Per-chip-mean is the more honest view of per-scene
# reliability -- it answers "how good is this model on a typical chip" rather
# than "how good is it on a typical water pixel". Reporting only one of them
# was the reason this project's earlier numbers could not be placed against the
# literature at all (spec section 15.1).
@dataclass
class AggregateMetrics:
    iou: float
    f1: float
    precision: float
    recall: float
    overall_accuracy: float
    kappa: float
    n_chips: int
    n_valid_pixels: int


def pool_counts(chip_metrics: list[ChipMetrics]) -> ConfusionCounts:
    """
    Sum the confusion counts of every chip into one combined count. Requires
    that each ChipMetrics carries its `counts` (compute_chip_metrics always
    sets it); a hand-constructed ChipMetrics without counts is an error rather
    than something to skip silently, because skipping would quietly change the
    denominator and produce a wrong-but-plausible aggregate.
    """
    if not chip_metrics:
        raise ValueError("cannot pool counts over an empty list of chip metrics")

    missing = [m.chip_id for m in chip_metrics if m.counts is None]
    if missing:
        raise ValueError(
            "cannot compute aggregate metrics: these chips have no confusion counts "
            f"(construct them via compute_chip_metrics): {missing[:5]}"
        )

    return ConfusionCounts(
        true_positive=sum(m.counts.true_positive for m in chip_metrics),
        false_positive=sum(m.counts.false_positive for m in chip_metrics),
        false_negative=sum(m.counts.false_negative for m in chip_metrics),
        true_negative=sum(m.counts.true_negative for m in chip_metrics),
    )


def aggregate_metrics(chip_metrics: list[ChipMetrics]) -> AggregateMetrics:
    """
    Pool every chip's confusion counts, then compute metrics once from the
    combined totals -- the dataset-level metric the literature reports.
    """
    counts = pool_counts(chip_metrics)
    return AggregateMetrics(
        iou=iou_score(counts),
        f1=f1_score(counts),
        precision=precision_score(counts),
        recall=recall_score(counts),
        overall_accuracy=overall_accuracy(counts),
        kappa=cohens_kappa(counts),
        n_chips=len(chip_metrics),
        n_valid_pixels=(
            counts.true_positive
            + counts.true_negative
            + counts.false_positive
            + counts.false_negative
        ),
    )


def summarize_per_event(chip_metrics: list[ChipMetrics]) -> dict[str, MetricSummary]:
    """
    Group per-chip metrics by flood event and summarize each group
    separately -- this is the per-event breakdown the spec calls out as
    non-optional: "averages can hide a model that only works on one event."
    """
    by_event: dict[str, list[ChipMetrics]] = {}
    for metric in chip_metrics:
        by_event.setdefault(metric.event, []).append(metric)

    return {event: summarize(metrics) for event, metrics in by_event.items()}
