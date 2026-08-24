"""
The shallow-ML baseline suite beyond the original 5-feature Random Forest
(benchmarks/random_forest.py, kept as-is and still reported as "RF ORIGINAL"
for comparison -- see benchmarks/ml_features.py's module docstring for why
its lack of denoising/texture/event-context was a real gap, not a design
choice).

Trains ExtraTrees, Random Forest, XGBoost, and LightGBM on the 15-feature set
(benchmarks/ml_features.py: 5 base + 6 texture + 4 event-relative channels)
and reports each. ExtraTrees turned out to be the best single model, by a
consistent margin over Random Forest, XGBoost, and LightGBM alike -- its
extra randomization (random split thresholds, not just random features) acts
as regularization against the speckle-noisy features here, and boosting
gained nothing over bagging on this problem. Same-family ensembling of these
four was tried and does NOT help (see benchmarks/cnn_results.md's ensembling
section for the contrasting CNN case, where cross-*architecture* ensembling
was the single best gain there): all four models share the same 15 features
and the same tree-ensemble inductive bias, so their errors are correlated
enough that soft-voting them together lands *below* ExtraTrees alone.

The decision threshold matters a great deal here and is NOT 0.5. Training
draws a class-balanced sample (see sample_training_pixels in
benchmarks/random_forest.py, reused here), so the model's implied prior is
50/50 water/non-water while real chips are ~12% water -- predicted
probabilities come out systematically inflated. The threshold is swept on
the train split it was never trained on... no: swept on VAL, exactly once,
never on test, then applied unchanged to test -- the same discipline
benchmarks/tune_otsu_hand.py already established for the Otsu+HAND
thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from benchmarks.metrics import (
    AggregateMetrics,
    ChipMetrics,
    MetricSummary,
    compute_chip_metrics,
    event_name,
    summarize,
)
from benchmarks.metrics import aggregate_metrics as _aggregate_metrics
from benchmarks.ml_features import (
    FEATURE_NAMES_FULL,
    build_full_features,
    fit_fill_values,
    impute_with_fill,
)
from data.chip_terrain import TerrainCache, get_terrain
from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER
from data.sen1floods11 import Sen1Floods11Dataset

DEFAULT_N_ESTIMATORS = 200
DEFAULT_MIN_SAMPLES_LEAF = 2
DEFAULT_SAMPLES_PER_CLASS_PER_CHIP = 1000

# Selected on VAL by benchmarks/tune_ml_ensemble.py -- see that script's
# output for the sweep this came from, and this module's docstring for why
# 0.5 is the wrong default here.
DEFAULT_DECISION_THRESHOLD = 0.75


def build_model_registry(seed: int) -> dict[str, object]:
    """
    The four tree-ensemble models this baseline compares, all seeded
    identically for a given run. XGBoost/LightGBM are imported lazily so a
    machine without them installed can still import this module and use
    ExtraTrees/RandomForest alone.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "rf": RandomForestClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            random_state=seed,
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            random_state=seed,
            n_jobs=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        ),
        "lgbm": LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def sample_pixels_sar_finite(
    features: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS_PER_CHIP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Class-balanced pixel sampling, same shape as
    benchmarks/random_forest.py's sample_training_pixels, but with a
    deliberately different finiteness check: only the SAR bands (the first 3
    feature columns -- VV_db, VH_db, VV_VH_ratio, per FEATURE_NAMES_FULL's
    order) must be finite for a pixel to be eligible, not every feature.

    This is the point of the whole module. The original function's
    all-columns-finite check means a pixel is dropped from training whenever
    ANY feature is NaN, including the terrain/texture columns -- and
    data/hand.py's NaN border is water-enriched (see benchmarks/otsu_hand.py's
    plausible_terrain_mask docstring for the measured numbers on the same
    underlying bug). Reusing that check here would silently reintroduce the
    exact problem this baseline exists to fix. Non-finite terrain/texture
    values are handled later by impute_with_fill, not by exclusion.
    """
    label_flat = label.reshape(-1)
    valid = (label_flat != LABEL_IGNORE) & np.isfinite(features[:, :3]).all(axis=1)

    sampled_features, sampled_labels = [], []
    for class_value in (LABEL_WATER, LABEL_NON_WATER):
        class_indices = np.flatnonzero(valid & (label_flat == class_value))
        if class_indices.size == 0:
            continue
        n = min(samples_per_class, class_indices.size)
        chosen = rng.choice(class_indices, size=n, replace=False)
        sampled_features.append(features[chosen])
        sampled_labels.append(label_flat[chosen])

    if not sampled_features:
        return (
            np.empty((0, features.shape[1]), dtype=features.dtype),
            np.empty((0,), dtype=label.dtype),
        )
    return np.concatenate(sampled_features), np.concatenate(sampled_labels)


def collect_training_pixels(
    dataset: Sen1Floods11Dataset,
    dem_dir: str | Path,
    event_stats: dict[str, dict[str, float]],
    seed: int,
    terrain_cache: TerrainCache | None = None,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS_PER_CHIP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full 15-feature table for every chip in `dataset` and draw a
    class-balanced pixel sample from each, same sampling function the
    original Random Forest baseline uses. `require_all_finite=False`: only
    the SAR bands must be real for a pixel to be eligible (terrain/texture
    NaN gets imputed downstream via impute_with_fill), matching this
    baseline's whole point of not vetoing the HAND border.
    """
    dem_dir = Path(dem_dir)
    rng = np.random.default_rng(seed)

    all_features, all_labels = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        slope, hand = get_terrain(sample.chip_id, dem_dir, terrain_cache)
        stats = event_stats[event_name(sample.chip_id)]

        features = build_full_features(sample.image, slope, hand, stats)
        sampled_features, sampled_labels = sample_pixels_sar_finite(
            features, sample.label, rng, samples_per_class=samples_per_class
        )
        if sampled_features.size:
            all_features.append(sampled_features)
            all_labels.append(sampled_labels)

    return np.concatenate(all_features), np.concatenate(all_labels)


def train_ml_ensemble(
    train_dataset: Sen1Floods11Dataset,
    dem_dir: str | Path,
    train_event_stats: dict[str, dict[str, float]],
    seed: int = 0,
    terrain_cache: TerrainCache | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """
    Train all four models in the registry on the same training pixel sample.
    Returns (models, fill) -- `fill` is the per-column median used to impute
    non-finite features, fit on this same training sample so it never sees
    val/test data.
    """
    features, labels = collect_training_pixels(
        train_dataset, dem_dir, train_event_stats, seed, terrain_cache
    )
    fill = fit_fill_values(features)
    imputed = impute_with_fill(features, fill)
    targets = (labels == LABEL_WATER).astype(np.int8)

    models = build_model_registry(seed)
    for model in models.values():
        model.fit(imputed, targets)
    return models, fill


PredictProbaFn = Callable[[np.ndarray], np.ndarray]


def predict_water_mask_ml(
    model_or_avg: object | list[object],
    features: np.ndarray,
    fill: np.ndarray,
    shape: tuple[int, int],
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> np.ndarray:
    """
    Predict a full water mask for one chip. A pixel is only scored if its SAR
    bands (first 3 feature columns) are finite -- a genuinely no-data pixel,
    not a border artifact -- and predicted non-water otherwise, the same
    no-data convention every other baseline in this project uses.

    `model_or_avg` is either a single fitted model or a list of them (for
    soft-voting an ensemble): predicted probabilities are averaged across the
    list before the threshold is applied.
    """
    models = model_or_avg if isinstance(model_or_avg, list) else [model_or_avg]

    predictions = np.zeros(features.shape[0], dtype=bool)
    valid = np.isfinite(features[:, :3]).all(axis=1)
    if valid.any():
        imputed = impute_with_fill(features[valid], fill)
        probs = np.mean([m.predict_proba(imputed)[:, 1] for m in models], axis=0)
        predictions[valid] = probs >= threshold
    return predictions.reshape(shape)


def evaluate_ml_baseline(
    model_or_avg: object | list[object],
    dataset: Sen1Floods11Dataset,
    dem_dir: str | Path,
    event_stats: dict[str, dict[str, float]],
    fill: np.ndarray,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    terrain_cache: TerrainCache | None = None,
) -> list[ChipMetrics]:
    """Score one model (or soft-voted list of models) over every chip in `dataset`."""
    dem_dir = Path(dem_dir)
    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        slope, hand = get_terrain(sample.chip_id, dem_dir, terrain_cache)
        stats = event_stats[event_name(sample.chip_id)]

        features = build_full_features(sample.image, slope, hand, stats)
        predicted = predict_water_mask_ml(
            model_or_avg, features, fill, sample.label.shape, threshold
        )
        results.append(compute_chip_metrics(sample.chip_id, predicted, sample.label))
    return results


def summarize_ml(chip_metrics: list[ChipMetrics]) -> tuple[AggregateMetrics, MetricSummary]:
    """Both the pooled/literature-comparable summary and the per-chip one, in one call."""
    return _aggregate_metrics(chip_metrics), summarize(chip_metrics)


__all__ = [
    "DEFAULT_DECISION_THRESHOLD",
    "FEATURE_NAMES_FULL",
    "build_model_registry",
    "collect_training_pixels",
    "evaluate_ml_baseline",
    "predict_water_mask_ml",
    "summarize_ml",
    "train_ml_ensemble",
]
