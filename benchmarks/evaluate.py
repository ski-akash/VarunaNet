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
    compute_chip_metrics,
    summarize,
    summarize_per_event,
)
from benchmarks.otsu import otsu_water_mask
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


def _otsu_or_all_non_water(compute: Callable[[], np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """
    otsu_water_mask -- and by extension otsu_hand_water_mask, which calls it
    internally -- deliberately raises ValueError when a chip's VV band is
    entirely NaN, since there's no histogram to threshold at all. This is a
    real edge case in the test split (Paraguay_34417 is entirely NaN in VV,
    a scene-edge no-data artifact, not a bug), so the
    harness catches it here and falls back to predicting the whole chip as
    non-water, the same convention benchmarks/random_forest.py already uses
    for individual NaN pixels.
    """
    try:
        return compute()
    except ValueError as error:
        if "no finite VV values" not in str(error):
            raise
        return np.zeros(shape, dtype=bool)


def otsu_predict(sample: Sen1Floods11Sample, slope: np.ndarray, hand: np.ndarray) -> np.ndarray:
    vv_db = sample.image[0]
    return _otsu_or_all_non_water(lambda: otsu_water_mask(vv_db), vv_db.shape)


def otsu_hand_predict(
    sample: Sen1Floods11Sample, slope: np.ndarray, hand: np.ndarray
) -> np.ndarray:
    vv_db = sample.image[0]
    return _otsu_or_all_non_water(lambda: otsu_hand_water_mask(vv_db, hand, slope), vv_db.shape)


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
    """Print an aggregate summary and a per-event breakdown for one baseline."""
    overall = summarize(chip_metrics)
    print(f"\n=== {name} ===")
    print(
        f"overall: mean IoU {overall.mean_iou:.3f}, median IoU {overall.median_iou:.3f}, "
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

    print_report("Otsu", evaluate_baseline(otsu_predict, test_dataset, DEM_DIR))
    print_report("Otsu + HAND", evaluate_baseline(otsu_hand_predict, test_dataset, DEM_DIR))

    rf_model = train_random_forest_baseline(train_dataset, DEM_DIR)
    rf_results = evaluate_baseline(make_random_forest_predict(rf_model), test_dataset, DEM_DIR)
    print_report("Random Forest", rf_results)
