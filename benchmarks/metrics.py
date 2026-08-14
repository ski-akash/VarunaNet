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


@dataclass
class ChipMetrics:
    chip_id: str
    event: str
    iou: float
    f1: float
    precision: float
    recall: float


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
