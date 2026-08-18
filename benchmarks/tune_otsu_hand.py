"""
Grid search over the Otsu+HAND terrain-plausibility thresholds
(benchmarks/otsu_hand.py's DEFAULT_HAND_THRESHOLD_M /
DEFAULT_SLOPE_THRESHOLD_DEG), which have never been anything but
untuned literature starting points. With those defaults, Otsu+HAND
(0.281 mean IoU) currently scores *worse* than plain Otsu (0.304) --
HAND/slope filtering can only ever remove candidate water pixels from
Otsu's output, never add any, so it's a pure precision-for-recall trade;
the untuned 15m/5deg cutoffs are evidently too aggressive for this data,
trimming real flood pixels in events where water spreads further from a
drainage channel than that.

Tuned against the TRAIN split, not test: picking thresholds by directly
maximizing IoU on the test split's own labels would be real test-set
leakage (a real deployment wouldn't have test labels to tune against
either, and this project's spec explicitly rejects results that aren't
honestly held-out). The winning threshold pair is then scored exactly
once against the untouched test split for the number that actually goes
in benchmarks/RESULTS.md.

Otsu's own candidate mask (VH band, denoised, per-event threshold --
see benchmarks/otsu.py) doesn't depend on either terrain parameter, so
it's computed once per chip and reused across every grid cell instead of
recomputed 42 times over.
"""

from pathlib import Path

import numpy as np

from benchmarks.evaluate import build_terrain_cache, compute_per_event_otsu_thresholds
from benchmarks.metrics import ChipMetrics, MetricSummary, compute_chip_metrics, event_name, summarize
from benchmarks.otsu import otsu_water_mask, smooth_backscatter
from benchmarks.otsu_hand import DEFAULT_HAND_THRESHOLD_M, DEFAULT_SLOPE_THRESHOLD_DEG
from data.sen1floods11 import Sen1Floods11Dataset

HAND_THRESHOLD_CANDIDATES = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
SLOPE_THRESHOLD_CANDIDATES = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]

CandidateEntry = tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def precompute_candidates(dataset: Sen1Floods11Dataset, dem_dir: Path) -> list[CandidateEntry]:
    """
    For every chip: (chip_id, label, otsu_candidate_water_mask, slope, hand).
    The candidate mask is independent of the HAND/slope grid search below,
    so it's computed once here rather than inside score_grid_cell.
    """
    event_thresholds = compute_per_event_otsu_thresholds(dataset)
    chip_ids = [dataset[i].chip_id for i in range(len(dataset))]
    terrain_cache = build_terrain_cache(chip_ids, dem_dir)

    entries: list[CandidateEntry] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        band = smooth_backscatter(sample.image[1])  # VH, same as benchmarks/evaluate.py
        threshold = event_thresholds.get(event_name(sample.chip_id))
        candidate = (
            np.zeros(band.shape, dtype=bool)
            if threshold is None
            else otsu_water_mask(band, threshold)
        )
        slope, hand = terrain_cache[sample.chip_id]
        entries.append((sample.chip_id, sample.label, candidate, slope, hand))
    return entries


def score_grid_cell(
    entries: list[CandidateEntry], hand_threshold: float, slope_threshold: float
) -> MetricSummary:
    chip_metrics: list[ChipMetrics] = []
    for chip_id, label, candidate, slope, hand in entries:
        plausible_terrain = (hand <= hand_threshold) & (slope <= slope_threshold)
        predicted = candidate & plausible_terrain
        chip_metrics.append(compute_chip_metrics(chip_id, predicted, label))
    return summarize(chip_metrics)


if __name__ == "__main__":
    DATA_ROOT = Path("datasets/sen1floods11")
    IMAGE_DIR = DATA_ROOT / "S1Hand"
    LABEL_DIR = DATA_ROOT / "LabelHand"
    DEM_DIR = DATA_ROOT / "DEMHand"
    SPLITS_DIR = DATA_ROOT / "splits"

    train_dataset = Sen1Floods11Dataset(IMAGE_DIR, LABEL_DIR, SPLITS_DIR / "flood_train_data.csv")
    test_dataset = Sen1Floods11Dataset(IMAGE_DIR, LABEL_DIR, SPLITS_DIR / "flood_test_data.csv")

    print("Precomputing Otsu candidate masks + terrain for the train split...")
    train_entries = precompute_candidates(train_dataset, DEM_DIR)

    print(
        f"Grid search: {len(HAND_THRESHOLD_CANDIDATES)}x{len(SLOPE_THRESHOLD_CANDIDATES)} "
        "combinations, scored on TRAIN only..."
    )
    best_iou, best_hand, best_slope = float("-inf"), DEFAULT_HAND_THRESHOLD_M, DEFAULT_SLOPE_THRESHOLD_DEG
    for hand_threshold in HAND_THRESHOLD_CANDIDATES:
        for slope_threshold in SLOPE_THRESHOLD_CANDIDATES:
            summary = score_grid_cell(train_entries, hand_threshold, slope_threshold)
            print(
                f"  hand<={hand_threshold:>6.1f}m slope<={slope_threshold:>5.1f}deg  "
                f"train mean_iou={summary.mean_iou:.4f}"
            )
            if summary.mean_iou > best_iou:
                best_iou, best_hand, best_slope = summary.mean_iou, hand_threshold, slope_threshold

    print(
        f"\nBest on train: hand_threshold={best_hand}m, slope_threshold={best_slope}deg "
        f"(train mean_iou={best_iou:.4f})"
    )

    print("\nScoring the chosen thresholds on the held-out TEST split (never used for selection)...")
    test_entries = precompute_candidates(test_dataset, DEM_DIR)

    tuned_summary = score_grid_cell(test_entries, best_hand, best_slope)
    print(
        f"tuned  (hand<={best_hand}m, slope<={best_slope}deg): "
        f"mean_iou={tuned_summary.mean_iou:.4f} median_iou={tuned_summary.median_iou:.4f} "
        f"mean_f1={tuned_summary.mean_f1:.4f} mean_precision={tuned_summary.mean_precision:.4f} "
        f"mean_recall={tuned_summary.mean_recall:.4f}"
    )

    default_summary = score_grid_cell(
        test_entries, DEFAULT_HAND_THRESHOLD_M, DEFAULT_SLOPE_THRESHOLD_DEG
    )
    print(
        f"untuned (hand<={DEFAULT_HAND_THRESHOLD_M}m, slope<={DEFAULT_SLOPE_THRESHOLD_DEG}deg): "
        f"mean_iou={default_summary.mean_iou:.4f} median_iou={default_summary.median_iou:.4f}"
    )
