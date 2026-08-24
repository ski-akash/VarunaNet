"""
Multi-seed evaluation of the ExtraTrees baseline (benchmarks/ml_ensemble.py):
reports mean +/- std over 3 seeds on the official test split, the same rigor
this project already holds its CNN results to (spec section 4.2: "report mean
+/- std over >=3 seeds for the headline model -- single-run numbers are not
evidence"). One seed's pooled IoU of 0.6421 looked strong; this is what
checks whether that number is real or a favorable draw, the same check that
caught speckle-noise augmentation's seed-1-only result evaporating across 3
seeds (see benchmarks/cnn_results.md).

Terrain (slope, HAND) and per-event statistics don't depend on the model seed,
so both are computed once per split and reused across all 3 seeds -- the same
caching principle benchmarks/hold_one_event_out.py already established for
terrain (recomputing pysheds flow routing 3x over would make that the
dominant cost of this script, not model training).

Run directly with `python -m benchmarks.evaluate_ml_ensemble`.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from benchmarks.ml_ensemble import (
    DEFAULT_DECISION_THRESHOLD,
    evaluate_ml_baseline,
    summarize_ml,
    train_ml_ensemble,
)
from benchmarks.ml_features import compute_event_stats
from data.chip_terrain import build_terrain_cache
from data.sen1floods11 import Sen1Floods11Dataset

SEEDS = (0, 1, 2)


def run(threshold: float = DEFAULT_DECISION_THRESHOLD) -> None:
    data_root = Path("datasets/sen1floods11")
    image_dir = data_root / "S1Hand"
    label_dir = data_root / "LabelHand"
    dem_dir = data_root / "DEMHand"
    splits_dir = data_root / "splits"

    train_dataset = Sen1Floods11Dataset(image_dir, label_dir, splits_dir / "flood_train_data.csv")
    test_dataset = Sen1Floods11Dataset(image_dir, label_dir, splits_dir / "flood_test_data.csv")

    print("computing per-event statistics (train, test)...")
    train_event_stats = compute_event_stats(train_dataset)
    test_event_stats = compute_event_stats(test_dataset)

    print("building terrain caches (train, test) -- one time, reused across all seeds...")
    train_chip_ids = [train_dataset[i].chip_id for i in range(len(train_dataset))]
    test_chip_ids = [test_dataset[i].chip_id for i in range(len(test_dataset))]
    train_terrain = build_terrain_cache(train_chip_ids, dem_dir)
    test_terrain = build_terrain_cache(test_chip_ids, dem_dir)

    test_pooled_ious, test_perchip_ious, test_oas, test_kappas = [], [], [], []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        models, fill = train_ml_ensemble(
            train_dataset, dem_dir, train_event_stats, seed=seed, terrain_cache=train_terrain
        )
        et_model = models["et"]

        test_metrics = evaluate_ml_baseline(
            et_model, test_dataset, dem_dir, test_event_stats, fill, threshold, test_terrain
        )
        pooled, per_chip = summarize_ml(test_metrics)
        print(
            f"  ExtraTrees, seed={seed}, thr={threshold}: "
            f"pooled IoU {pooled.iou:.4f}, per-chip mean IoU {per_chip.mean_iou:.4f}, "
            f"OA {pooled.overall_accuracy:.4f}, kappa {pooled.kappa:.4f}"
        )
        test_pooled_ious.append(pooled.iou)
        test_perchip_ious.append(per_chip.mean_iou)
        test_oas.append(pooled.overall_accuracy)
        test_kappas.append(pooled.kappa)

    seed_rounded = [round(v, 4) for v in test_pooled_ious]
    print(f"\n=== ExtraTrees, {len(SEEDS)}-seed summary, test split, threshold={threshold} ===")
    print(
        f"pooled IoU:   mean {statistics.mean(test_pooled_ious):.4f} "
        f"+/- {statistics.stdev(test_pooled_ious):.4f}  (seeds: {seed_rounded})"
    )
    print(
        f"per-chip IoU: mean {statistics.mean(test_perchip_ious):.4f} "
        f"+/- {statistics.stdev(test_perchip_ious):.4f}"
    )
    print(
        f"OA:           mean {statistics.mean(test_oas):.4f} "
        f"+/- {statistics.stdev(test_oas):.4f}"
    )
    print(
        f"kappa:        mean {statistics.mean(test_kappas):.4f} "
        f"+/- {statistics.stdev(test_kappas):.4f}"
    )


if __name__ == "__main__":
    run()
