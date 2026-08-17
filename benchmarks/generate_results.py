"""
Regenerates benchmarks/RESULTS.md from a single command (`make bench`).

Runs every classical/ML baseline (Otsu, Otsu+HAND, Random Forest) two
ways -- once on the official train/val/test split, once under
hold-one-event-out cross-validation -- and writes both result tables,
their per-event breakdowns, and a short interpretation of each to
benchmarks/RESULTS.md. Per the project rule stated in benchmarks/README.md:
no result goes in the README without the baseline it beat and the metric
used, so this always reports every baseline side by side rather than one
model's number in isolation.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.evaluate import (
    build_terrain_cache,
    compute_per_event_otsu_thresholds,
    evaluate_baseline,
    make_otsu_hand_predict,
    make_otsu_predict,
    make_random_forest_predict,
    train_random_forest_baseline,
)
from benchmarks.hold_one_event_out import run_hold_one_event_out
from benchmarks.metrics import ChipMetrics, MetricSummary, summarize, summarize_per_event
from data.sen1floods11 import Sen1Floods11Dataset


def run_official_split(
    image_dir: Path, label_dir: Path, dem_dir: Path, splits_dir: Path
) -> dict[str, list[ChipMetrics]]:
    """Train (where applicable) on the official train split, score on the official test split."""
    train_dataset = Sen1Floods11Dataset(image_dir, label_dir, splits_dir / "flood_train_data.csv")
    test_dataset = Sen1Floods11Dataset(image_dir, label_dir, splits_dir / "flood_test_data.csv")

    chip_ids = [
        image_filename.split("_S1Hand")[0]
        for image_filename, _ in train_dataset.pairs + test_dataset.pairs
    ]
    terrain_cache = build_terrain_cache(chip_ids, dem_dir)

    # Thresholds come from the test set's own chips, pooled per event --
    # see benchmarks/evaluate.py's compute_per_event_otsu_thresholds.
    event_thresholds = compute_per_event_otsu_thresholds(test_dataset)
    results = {
        "Otsu": evaluate_baseline(
            make_otsu_predict(event_thresholds), test_dataset, dem_dir, terrain_cache
        ),
        "Otsu + HAND": evaluate_baseline(
            make_otsu_hand_predict(event_thresholds), test_dataset, dem_dir, terrain_cache
        ),
    }
    rf_model = train_random_forest_baseline(train_dataset, dem_dir, terrain_cache=terrain_cache)
    results["Random Forest"] = evaluate_baseline(
        make_random_forest_predict(rf_model), test_dataset, dem_dir, terrain_cache
    )
    return results


def _summary_table(summaries: dict[str, MetricSummary]) -> str:
    header = "| Model | Mean IoU | Median IoU | Mean F1 | Mean Precision | Mean Recall | n |"
    separator = "|---|---|---|---|---|---|---|"
    rows = [
        f"| {name} | {s.mean_iou:.3f} | {s.median_iou:.3f} | {s.mean_f1:.3f} "
        f"| {s.mean_precision:.3f} | {s.mean_recall:.3f} | {s.n_chips} |"
        for name, s in summaries.items()
    ]
    return "\n".join([header, separator, *rows])


def _per_event_table(per_baseline_events: dict[str, dict[str, MetricSummary]]) -> str:
    event_names = sorted({event for events in per_baseline_events.values() for event in events})
    baseline_names = list(per_baseline_events.keys())

    header = "| Event | " + " | ".join(f"{name} IoU" for name in baseline_names) + " | n |"
    separator = "|---" * (len(baseline_names) + 2) + "|"

    rows = []
    for event in event_names:
        cells = []
        n_chips = None
        for name in baseline_names:
            summary = per_baseline_events[name].get(event)
            cells.append(f"{summary.mean_iou:.3f}" if summary else "-")
            n_chips = summary.n_chips if summary else n_chips
        rows.append(f"| {event} | " + " | ".join(cells) + f" | {n_chips} |")

    return "\n".join([header, separator, *rows])


def _best_baseline(summaries: dict[str, MetricSummary]) -> str:
    return max(summaries, key=lambda name: summaries[name].mean_iou)


def _hardest_and_easiest_events(per_event: dict[str, MetricSummary]) -> tuple[str, str]:
    hardest = min(per_event, key=lambda event: per_event[event].mean_iou)
    easiest = max(per_event, key=lambda event: per_event[event].mean_iou)
    return hardest, easiest


def _describe_extreme_event(
    event: str, per_baseline_events: dict[str, dict[str, MetricSummary]], pick_hardest: bool
) -> str:
    """
    Describe whether `event` (the best baseline's hardest/easiest event) is
    also the hardest/easiest for every other baseline, or just for the best
    one -- checking this instead of assuming it avoids overclaiming
    "consistent across every baseline" when only one baseline actually
    agrees.
    """
    per_baseline_extreme = {
        name: (min if pick_hardest else max)(events, key=lambda e: events[e].mean_iou)
        for name, events in per_baseline_events.items()
    }
    if len(set(per_baseline_extreme.values())) == 1:
        return "every baseline"
    agreeing = sorted(name for name, extreme in per_baseline_extreme.items() if extreme == event)
    return f"{', '.join(agreeing)} only -- other baselines disagree"


def render_report(
    official_summaries: dict[str, MetricSummary],
    official_per_event: dict[str, dict[str, MetricSummary]],
    hoeo_summaries: dict[str, MetricSummary],
    hoeo_per_event: dict[str, dict[str, MetricSummary]],
) -> str:
    best_official = _best_baseline(official_summaries)
    best_hoeo = _best_baseline(hoeo_summaries)
    hardest_event, easiest_event = _hardest_and_easiest_events(hoeo_per_event[best_hoeo])
    hardest_scope = _describe_extreme_event(hardest_event, hoeo_per_event, pick_hardest=True)
    easiest_scope = _describe_extreme_event(easiest_event, hoeo_per_event, pick_hardest=False)

    official_mean = official_summaries[best_hoeo].mean_iou
    hoeo_mean = hoeo_summaries[best_hoeo].mean_iou
    generalization_gap = official_mean - hoeo_mean

    return f"""# Benchmark Results

Regenerated by `make bench` (`benchmarks/generate_results.py`) -- do not edit by hand,
rerun `make bench` instead. Every model here is a classical or shallow-ML baseline
(Phase 2 of the project); the CNN and transformer models that later results will be
benchmarked against come in Phase 3+.

## Official train/val/test split

Trained (where applicable) on the official train split, scored on the official test
split.

{_summary_table(official_summaries)}

### Per-event breakdown

{_per_event_table(official_per_event)}

**Interpretation:** {best_official} has the best mean IoU on the official split. Mean
and median diverge noticeably for every baseline here (e.g. Otsu's mean sits well above
its median), which is the signature of a model that does fine on most chips but badly
on a few outliers -- exactly why this project reports both, not just the mean.

## Hold-one-event-out cross-validation

Each of the 11 flood events is held out in turn: Random Forest trains on the other 10
events only and is tested on the held-out one (Otsu and Otsu+HAND don't train, so
they're scored fold-by-fold for a like-for-like comparison). This is the harder,
more honest generalization test: the official split mixes chips from every event into
both train and test, so a model can partly succeed by fitting an event's specific
terrain and backscatter rather than truly generalizing to conditions it has never
seen.

{_summary_table(hoeo_summaries)}

### Per-event breakdown

{_per_event_table(hoeo_per_event)}

**Interpretation:** {best_hoeo} still wins under hold-one-event-out, but its mean IoU
drops from {official_mean:.3f} (official split) to {hoeo_mean:.3f} (unseen-event
average) -- a generalization gap of {generalization_gap:.3f}, the real cost of testing
on an event the model never trained on. {easiest_event} is the easiest event for
{easiest_scope}, even though it never appeared in the official test split -- new
information this split alone would have missed. {hardest_event} is the hardest event
for {hardest_scope}.
"""


if __name__ == "__main__":
    DATA_ROOT = Path("datasets/sen1floods11")
    IMAGE_DIR = DATA_ROOT / "S1Hand"
    LABEL_DIR = DATA_ROOT / "LabelHand"
    DEM_DIR = DATA_ROOT / "DEMHand"
    SPLITS_DIR = DATA_ROOT / "splits"

    print("Running official train/val/test split evaluation...")
    official_results = run_official_split(IMAGE_DIR, LABEL_DIR, DEM_DIR, SPLITS_DIR)
    official_summaries = {name: summarize(m) for name, m in official_results.items()}
    official_per_event = {name: summarize_per_event(m) for name, m in official_results.items()}

    print("Running hold-one-event-out cross-validation...")
    hoeo_results = run_hold_one_event_out(IMAGE_DIR, LABEL_DIR, DEM_DIR, SPLITS_DIR)
    hoeo_summaries = {name: summarize(m) for name, m in hoeo_results.items()}
    hoeo_per_event = {name: summarize_per_event(m) for name, m in hoeo_results.items()}

    report = render_report(official_summaries, official_per_event, hoeo_summaries, hoeo_per_event)
    output_path = Path("benchmarks/RESULTS.md")
    output_path.write_text(report)
    print(f"Wrote {output_path}")
