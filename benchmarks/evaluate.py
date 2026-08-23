"""
Wires the three baselines (Otsu, Otsu+HAND, Random Forest) through the
metrics module (benchmarks/metrics.py) on real Sen1Floods11 chips, so all
three get scored by the same code path and produce real per-chip and
per-event numbers -- replacing the throwaway one-off scripts each baseline
was originally validated with.

Hold-one-event-out cross-validation and a `make bench` command that
regenerates benchmarks/RESULTS.md are both built on top of this module's
evaluate_baseline()/train_random_forest_baseline() functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from benchmarks.metrics import (
    ChipMetrics,
    MetricSummary,
    aggregate_metrics,
    compute_chip_metrics,
    event_name,
    summarize,
    summarize_per_event,
)
from benchmarks.otsu import compute_otsu_threshold, otsu_water_mask, smooth_backscatter
from benchmarks.otsu_hand import otsu_hand_water_mask
from benchmarks.random_forest import (
    build_pixel_features,
    predict_water_mask,
    sample_training_pixels,
    train_random_forest,
)
from data.chip_terrain import (
    CHIP_PIXEL_SIZE_M,
    HAND_ACCUMULATION_THRESHOLD,
    TerrainCache,
    build_terrain_cache,
    compute_terrain_layers,
    get_terrain,
    load_dem,
)
from data.sen1floods11 import Sen1Floods11Dataset, Sen1Floods11Sample

# Re-exported from data.chip_terrain (see that module, shared with
# training/sen1floods11_dataset.py) so this file's own imports above stay
# the public names existing callers/tests already use. Note for anyone
# patching HAND_ACCUMULATION_THRESHOLD in a test: patch it on
# data.chip_terrain, not on this module -- compute_terrain_layers is
# *defined* there, so it reads the module-global at call time from that
# module's own namespace, not from whatever name this import rebinds here.
__all__ = [
    "CHIP_PIXEL_SIZE_M",
    "HAND_ACCUMULATION_THRESHOLD",
    "TerrainCache",
    "build_terrain_cache",
    "compute_terrain_layers",
    "load_dem",
]


# Every baseline is scored through the same evaluate_baseline() loop below,
# so each one is exposed as a function of this one shape: given a chip and
# its terrain layers, return a boolean water mask. This is what lets Otsu,
# Otsu+HAND, and a trained Random Forest all plug into identical evaluation
# code instead of three near-duplicate scoring scripts.
PredictFn = Callable[[Sen1Floods11Sample, np.ndarray, np.ndarray], np.ndarray]


# sample.image is [VV_db, VH_db, VV_VH_ratio] (data/sen1floods11.py).
# VH, not VV, is what this project's Otsu baseline actually thresholds --
# confirmed against real evidence (see benchmarks/otsu.py's module
# docstring), not the original assumption that VV (the more commonly
# cited band for water detection in SAR literature generally) was right
# for this specific dataset's published baseline.
_OTSU_BAND_INDEX = 1


def _smoothed_otsu_band(sample: Sen1Floods11Sample) -> np.ndarray:
    return smooth_backscatter(sample.image[_OTSU_BAND_INDEX])


def compute_per_event_otsu_thresholds(dataset: Sen1Floods11Dataset) -> dict[str, float]:
    """
    Otsu's method needs a real amount of bimodal signal (dark water vs.
    bright land) to find a good split; a single 512x512 chip often
    doesn't have enough of both classes in it to do that well -- most
    chips are mostly-dry. The published Sen1Floods11 baseline numbers
    (Bonafilia et al., CVPRW 2020) threshold Otsu per *flood event*, not
    per individual chip: every chip from the same event shares an
    acquisition and noise floor, so pooling them gives Otsu a real
    histogram to work with. Confirmed directly against this project's own
    data that doing the same thing here recovers a real chunk of the gap
    to the published number (official test split, per-chip VV: mean IoU
    0.211 -> per-event VV: 0.224 -> per-event VH+denoised: 0.27+, see
    benchmarks/otsu.py's module docstring for the full chain of evidence).

    Pools every finite (post-denoising) pixel across all of an event's
    chips within `dataset` and computes one Otsu threshold per event. An
    event whose every chip is entirely NaN (no finite pixels anywhere)
    simply gets no entry -- make_otsu_predict/make_otsu_hand_predict fall
    back to "no water anywhere" for those chips, the same as a single NaN
    chip would if asked to threshold itself from nothing.
    """
    pixels_by_event: dict[str, list[np.ndarray]] = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        band = _smoothed_otsu_band(sample)
        finite = band[np.isfinite(band)]
        if finite.size > 0:
            pixels_by_event.setdefault(event_name(sample.chip_id), []).append(finite)

    return {
        event: compute_otsu_threshold(np.concatenate(chunks))
        for event, chunks in pixels_by_event.items()
    }


def make_otsu_predict(event_thresholds: dict[str, float]) -> PredictFn:
    """
    Binds a per-event threshold map (see compute_per_event_otsu_thresholds
    above) into a PredictFn. A chip whose event has no entry (every chip
    in that event was entirely NaN) predicts no water anywhere.
    """

    def predict(sample: Sen1Floods11Sample, slope: np.ndarray, hand: np.ndarray) -> np.ndarray:
        band = _smoothed_otsu_band(sample)
        threshold = event_thresholds.get(event_name(sample.chip_id))
        if threshold is None:
            return np.zeros(band.shape, dtype=bool)
        return otsu_water_mask(band, threshold)

    return predict


def make_otsu_hand_predict(event_thresholds: dict[str, float]) -> PredictFn:
    """Same per-event threshold binding as make_otsu_predict, refined with HAND/slope."""

    def predict(sample: Sen1Floods11Sample, slope: np.ndarray, hand: np.ndarray) -> np.ndarray:
        band = _smoothed_otsu_band(sample)
        threshold = event_thresholds.get(event_name(sample.chip_id))
        if threshold is None:
            return np.zeros(band.shape, dtype=bool)
        return otsu_hand_water_mask(band, hand, slope, otsu_threshold=threshold)

    return predict


def all_dry_predict(sample: Sen1Floods11Sample, slope: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """
    The degenerate control: predict that nothing is water, anywhere, ever.

    This exists to make a specific point measurable rather than rhetorical.
    The published Assam/Brahmaputra SAR flood literature reports Overall
    Accuracy (93.6%/95.15% for the change-detection study, >82% for the
    RF/SVM/CART study) rather than IoU. Because flood water is a small
    minority of pixels, a model that detects nothing still scores a high OA.
    Running this baseline through the identical evaluation path as every real
    model quantifies exactly how much of a published OA figure is available
    for free -- which is the evidence behind this project's choice to report
    IoU as the primary metric (spec section 15.0/15.1).

    Its IoU is 0 (or NaN on a genuinely dry chip, where predicting no water is
    correct and the metric is undefined rather than wrong) and its Kappa is
    ~0, which is precisely the contrast worth showing.
    """
    return np.zeros(sample.label.shape, dtype=bool)


def make_random_forest_predict(model: RandomForestClassifier) -> PredictFn:
    """Bind a trained model into a PredictFn, matching the other baselines' signature."""

    def predict(sample: Sen1Floods11Sample, slope: np.ndarray, hand: np.ndarray) -> np.ndarray:
        features = build_pixel_features(sample.image, slope, hand)
        return predict_water_mask(model, features, shape=sample.label.shape)

    return predict


def evaluate_baseline(
    predict_fn: PredictFn,
    dataset: Sen1Floods11Dataset,
    dem_dir: str | Path,
    terrain_cache: TerrainCache | None = None,
) -> list[ChipMetrics]:
    """
    Run one baseline's predict function over every chip in `dataset`,
    scoring each prediction against its label via benchmarks/metrics.py.
    Shared by every baseline so a difference in the numbers reflects a real
    difference in the models, not a difference in how they were evaluated.
    """
    dem_dir = Path(dem_dir)
    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        slope, hand = get_terrain(sample.chip_id, dem_dir, terrain_cache)

        predicted = predict_fn(sample, slope, hand)
        results.append(compute_chip_metrics(sample.chip_id, predicted, sample.label))
    return results


def train_random_forest_baseline(
    train_dataset: Sen1Floods11Dataset,
    dem_dir: str | Path,
    seed: int = 0,
    terrain_cache: TerrainCache | None = None,
) -> RandomForestClassifier:
    """
    Train the Random Forest baseline on every chip in `train_dataset`,
    class-balanced sampling per chip (see benchmarks/random_forest.py).
    """
    dem_dir = Path(dem_dir)
    rng = np.random.default_rng(seed)

    all_features = []
    all_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        slope, hand = get_terrain(sample.chip_id, dem_dir, terrain_cache)

        features = build_pixel_features(sample.image, slope, hand)
        sampled_features, sampled_labels = sample_training_pixels(features, sample.label, rng)
        all_features.append(sampled_features)
        all_labels.append(sampled_labels)

    training_features = np.concatenate(all_features)
    training_labels = np.concatenate(all_labels)
    return train_random_forest(training_features, training_labels, seed=seed)


def print_report(name: str, chip_metrics: list[ChipMetrics]) -> None:
    """
    Print both summaries and a per-event breakdown for one baseline.

    Two overall lines, not one, and the order matters: the pooled/aggregate
    line comes first because it is the number comparable to published work
    (Sen1Floods11's own figures, and the OA the Assam literature reports),
    while the per-chip-mean line below it is the more conservative per-scene
    view. Showing only one of them is what previously made this project's
    numbers impossible to place against the literature -- see the comment
    above aggregate_metrics() in benchmarks/metrics.py for why they differ.
    """
    pooled = aggregate_metrics(chip_metrics)
    overall = summarize(chip_metrics)
    print(f"\n=== {name} ===")
    print(
        f"pooled:   IoU {pooled.iou:.3f}, F1 {pooled.f1:.3f}, "
        f"precision {pooled.precision:.3f}, recall {pooled.recall:.3f}, "
        f"OA {pooled.overall_accuracy:.4f}, kappa {pooled.kappa:.3f} "
        f"(n={pooled.n_chips} chips, {pooled.n_valid_pixels:,} valid px)"
    )
    print(
        f"per-chip: mean IoU {overall.mean_iou:.3f}, median IoU {overall.median_iou:.3f}, "
        f"mean F1 {overall.mean_f1:.3f} (n={overall.n_chips})"
    )

    per_event = summarize_per_event(chip_metrics)
    for event in sorted(per_event):
        summary: MetricSummary = per_event[event]
        print(
            f"  {event:12s} mean IoU {summary.mean_iou:.3f}, median IoU {summary.median_iou:.3f} "
            f"(n={summary.n_chips})"
        )


if __name__ == "__main__":
    DATA_ROOT = Path("datasets/sen1floods11")
    IMAGE_DIR = DATA_ROOT / "S1Hand"
    LABEL_DIR = DATA_ROOT / "LabelHand"
    DEM_DIR = DATA_ROOT / "DEMHand"
    SPLITS_DIR = DATA_ROOT / "splits"

    train_dataset = Sen1Floods11Dataset(IMAGE_DIR, LABEL_DIR, SPLITS_DIR / "flood_train_data.csv")
    test_dataset = Sen1Floods11Dataset(IMAGE_DIR, LABEL_DIR, SPLITS_DIR / "flood_test_data.csv")

    # The degenerate control runs first, deliberately: every real baseline's
    # Overall Accuracy below should be read against this one's, since that is
    # the score available for predicting nothing at all.
    print_report("All-dry (control)", evaluate_baseline(all_dry_predict, test_dataset, DEM_DIR))

    # Thresholds come from the test set's own chips, same as the original
    # per-chip version did (Otsu doesn't train -- there's no separate fit
    # step) -- just pooled per event now instead of computed fresh per chip.
    event_thresholds = compute_per_event_otsu_thresholds(test_dataset)
    print_report(
        "Otsu", evaluate_baseline(make_otsu_predict(event_thresholds), test_dataset, DEM_DIR)
    )
    print_report(
        "Otsu + HAND",
        evaluate_baseline(make_otsu_hand_predict(event_thresholds), test_dataset, DEM_DIR),
    )

    rf_model = train_random_forest_baseline(train_dataset, DEM_DIR)
    rf_results = evaluate_baseline(make_random_forest_predict(rf_model), test_dataset, DEM_DIR)
    print_report("Random Forest", rf_results)
