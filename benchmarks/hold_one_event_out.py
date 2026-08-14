"""
Hold-one-event-out cross-validation.

The official train/val/test split mixes chips from all 11 Sen1Floods11
flood events into every split, so scoring against the official test
split partly measures "did the model memorize this event's terrain and
backscatter" rather than "can this model handle an event it has never
seen." Generalization to a genuinely unseen event is the harder, more
meaningful question -- for each of the 11 events in turn, this holds
that entire event out, trains (for Random Forest; Otsu and Otsu+HAND
don't train) on the other 10 events' chips, and tests only on the held-
out event's chips. Every chip is held out exactly once, so pooling the
per-chip metrics across all 11 folds and grouping by event (via
benchmarks/metrics.py's summarize_per_event) gives a true "never saw
this event during training" score for every event, for every baseline.

Terrain (slope + HAND) is computed once per chip and cached across
folds -- see evaluate.py's TerrainCache -- since recomputing it per fold
would mean recomputing the same chip's HAND up to 10 times over (once
per fold where that chip is in the training set).
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.evaluate import (
    build_terrain_cache,
    evaluate_baseline,
    make_random_forest_predict,
    otsu_hand_predict,
    otsu_predict,
    print_report,
    train_random_forest_baseline,
)
from benchmarks.metrics import ChipMetrics, event_name
from data.sen1floods11 import Sen1Floods11Dataset, read_split

# The three official split files together cover every hand-labeled chip;
# hold-one-event-out repartitions by event instead, so it needs the full
# pool to draw from, not just one split.
SPLIT_FILENAMES = ("flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv")


def load_all_pairs(splits_dir: str | Path) -> list[tuple[str, str]]:
    """Pool (image_filename, label_filename) pairs across all three official splits."""
    splits_dir = Path(splits_dir)
    pairs: list[tuple[str, str]] = []
    for filename in SPLIT_FILENAMES:
        pairs.extend(read_split(splits_dir / filename))
    return pairs


def group_pairs_by_event(pairs: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Bucket chip pairs by the flood event parsed from their chip id (see metrics.event_name)."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for image_filename, label_filename in pairs:
        chip_id = image_filename.split("_S1Hand")[0]
        groups.setdefault(event_name(chip_id), []).append((image_filename, label_filename))
    return groups


def run_hold_one_event_out(
    image_dir: str | Path,
    label_dir: str | Path,
    dem_dir: str | Path,
    splits_dir: str | Path,
    seed: int = 0,
) -> dict[str, list[ChipMetrics]]:
    """
    Run all three baselines through hold-one-event-out CV and return each
    baseline's per-chip metrics, pooled across every fold. Every chip
    appears exactly once, scored by a model (Random Forest) that never
    saw its own event during training -- Otsu and Otsu+HAND don't train,
    so holding an event out doesn't change their predictions, but they're
    still scored fold-by-fold so every baseline is compared on the same
    event partitioning.
    """
    all_pairs = load_all_pairs(splits_dir)
    groups = group_pairs_by_event(all_pairs)

    chip_ids = [image_filename.split("_S1Hand")[0] for image_filename, _ in all_pairs]
    terrain_cache = build_terrain_cache(chip_ids, dem_dir)

    results: dict[str, list[ChipMetrics]] = {"Otsu": [], "Otsu + HAND": [], "Random Forest": []}

    for held_out_event in sorted(groups):
        test_pairs = groups[held_out_event]
        train_pairs = [
            pair for event, pairs in groups.items() if event != held_out_event for pair in pairs
        ]

        test_dataset = Sen1Floods11Dataset.from_pairs(image_dir, label_dir, test_pairs)
        train_dataset = Sen1Floods11Dataset.from_pairs(image_dir, label_dir, train_pairs)

        results["Otsu"].extend(
            evaluate_baseline(otsu_predict, test_dataset, dem_dir, terrain_cache)
        )
        results["Otsu + HAND"].extend(
            evaluate_baseline(otsu_hand_predict, test_dataset, dem_dir, terrain_cache)
        )

        rf_model = train_random_forest_baseline(
            train_dataset, dem_dir, seed=seed, terrain_cache=terrain_cache
        )
        results["Random Forest"].extend(
            evaluate_baseline(
                make_random_forest_predict(rf_model), test_dataset, dem_dir, terrain_cache
            )
        )

        print(f"[hold-one-event-out] finished held-out event: {held_out_event}")

    return results


if __name__ == "__main__":
    DATA_ROOT = Path("datasets/sen1floods11")

    results = run_hold_one_event_out(
        image_dir=DATA_ROOT / "S1Hand",
        label_dir=DATA_ROOT / "LabelHand",
        dem_dir=DATA_ROOT / "DEMHand",
        splits_dir=DATA_ROOT / "splits",
    )

    for name, chip_metrics in results.items():
        print_report(name, chip_metrics)
