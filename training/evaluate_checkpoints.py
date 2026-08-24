"""
Re-scores every architecture/ablation checkpoint recorded in
benchmarks/cnn_results.md through the pooled IoU / per-chip IoU / OA / kappa
metrics benchmarks/evaluate.py's classical baselines and the shallow-ML
baselines (benchmarks/ml_ensemble.py) already report -- the remaining Step 0
work spec section 15.1 calls out explicitly: "re-score the Phase 3/4 CNN
checkpoints ... to get pooled IoU/OA/Kappa for every architecture in
cnn_results.md". Every number cnn_results.md has recorded so far is per-chip
only, so none of it is directly comparable to the classical/ML baselines or
to Sen1Floods11's own published figures until this runs.

Not a Hydra entrypoint like training/evaluate_test.py, deliberately. Each
checkpoint already carries its own full training config (architecture,
encoder, in_channels, dataset channel_indices, include_jrc_baseline --
training/checkpoint.py's save_checkpoint persists it precisely so a run's
provenance is self-contained), so re-deriving that from a Hydra config group
would risk silently mismatching a checkpoint against the wrong config --
peeking the checkpoint's own stored config is authoritative and cannot drift.

Terrain (slope, HAND) does not depend on architecture, channel selection, or
which checkpoint is being scored -- every dataset config first builds the
full 5-channel tensor, then channel_indices slices it (see
training/sen1floods11_dataset.py). So the terrain cache is built exactly
once here and reused across every checkpoint, instead of every checkpoint
paying pysheds's ~0.6-0.7s/chip flow-routing cost independently: one U-Net
evaluation alone measured ~3.5 minutes dominated by terrain recomputation
against a same-order-of-magnitude-smaller model forward pass. Re-deriving it
per checkpoint for ~24 checkpoints would cost over an hour of redundant CPU
work for identical numbers; sharing it turns the whole sweep into one GPU
allocation instead of ~24 short ones queuing independently.

Run directly on a GPU node (see training/evaluate_checkpoints_a100.sh for
the sbatch wrapper):
    python -m training.evaluate_checkpoints
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmarks.metrics import AggregateMetrics, MetricSummary, aggregate_metrics, summarize
from data.chip_terrain import TerrainCache, build_terrain_cache
from data.normalization import NormalizationStats
from data.sen1floods11 import read_split
from models.build_model import build_model
from training.evaluate_test import predict_chip_metrics
from training.sen1floods11_dataset import Sen1Floods11TorchDataset
from training.train import resolve_device

DATA_ROOT = Path("datasets/sen1floods11")
TEST_SPLIT_CSV = "flood_test_data.csv"
NORMALIZATION_STATS_PATH = "data/sen1floods11_normalization_stats.json"
JOB_RESULTS_ROOT = Path("/userhome/mtech/akashc1005/job_results")


@dataclass
class CheckpointEntry:
    label: str
    job_id: int
    tta: bool = False

    def checkpoint_path(self) -> Path:
        return JOB_RESULTS_ROOT / str(self.job_id) / "checkpoints" / "best.pt"


# Every checkpoint benchmarks/cnn_results.md currently reports a number for,
# taken from the exact job IDs each training/evaluate_*.sh script already
# points at -- not re-guessed. TTA rows reuse the SAME underlying checkpoint
# as their non-TTA counterpart (4-way flip test-time augmentation is an
# eval-time choice, not a different trained model), so they appear twice
# with the same job_id and tta=True/False, matching cnn_results.md's own
# "+TTA" rows.
#
# Deliberately excluded: the early/interim seed-1 checkpoints superseded by
# later, fixed runs (job IDs 1620/1629/1632/1648/1650/1656/1685 -- p100/v2
# smoke-test and finetuning-experiment runs cnn_results.md's actual reported
# table does not cite), and the prediction-level ensembles
# (training/evaluate_ensemble.py, training/evaluate_cross_ensemble.py),
# which combine several of these checkpoints' logits and need their own
# re-scoring pass on top of this one, not instead of it.
CHECKPOINTS: list[CheckpointEntry] = [
    CheckpointEntry("U-Net (ResNet-34), seed 1", 1634),
    CheckpointEntry("U-Net (ResNet-34), seed 1 + TTA", 1634, tta=True),
    CheckpointEntry("U-Net (ResNet-34), seed 2", 1654),
    CheckpointEntry("U-Net (ResNet-34), seed 3", 1660),
    CheckpointEntry("U-Net++ (ResNet-34)", 1658),
    CheckpointEntry("U-Net++ (ResNet-34) + TTA", 1658, tta=True),
    CheckpointEntry("DeepLabV3+ (ResNet-34)", 1667),
    CheckpointEntry("SegFormer-B0", 1670),
    CheckpointEntry("SegFormer-B2", 1673),
    CheckpointEntry("SegFormer-B2 + TTA", 1673, tta=True),
    CheckpointEntry("ChangeAwareUNet, seed 1", 1700),
    CheckpointEntry("ChangeAwareUNet, seed 2", 1714),
    CheckpointEntry("ChangeAwareUNet, seed 3", 1715),
    CheckpointEntry("VV+VH only, seed 1", 1664),
    CheckpointEntry("VV+VH only, seed 2", 1675),
    CheckpointEntry("VV+VH only, seed 3", 1686),
    CheckpointEntry("No ratio (drop VV_VH_ratio)", 1708),
    CheckpointEntry("No slope", 1710),
    CheckpointEntry("No HAND", 1712),
    CheckpointEntry("U-Net (ResNet-50)", 1786),
    CheckpointEntry("U-Net (ResNet-34), Focal loss", 1789),
    CheckpointEntry("Speckle, looks=1", 1731),
    CheckpointEntry("Speckle, looks=4, seed 1", 1732),
    CheckpointEntry("Speckle, looks=4, seed 2", 1753),
    CheckpointEntry("Speckle, looks=4, seed 3", 1757),
    CheckpointEntry("Speckle, looks=10", 1733),
    CheckpointEntry("U-Net (ResNet-18)", 1941),
    CheckpointEntry("U-Net (MobileNetV3-Large)", 1942),
]


@dataclass
class ScoredCheckpoint:
    entry: CheckpointEntry
    pooled: AggregateMetrics | None
    per_chip: MetricSummary | None
    seconds: float
    error: str | None = None


def _load_model_for_eval(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    """
    Load a checkpoint for scoring only -- model weights and its own stored
    config, nothing else. Deliberately does NOT go through
    training/checkpoint.py's load_checkpoint: that also restores
    optimizer/scheduler/RNG state, which matters for resuming training but
    is irrelevant (and, before a real bug fix, was actively broken on GPU --
    see training/checkpoint.py's _restore_rng_state docstring) for a
    read-only forward pass. encoder_weights=None skips downloading ImageNet
    weights that load_state_dict is about to overwrite anyway.

    model_cfg.get("architecture", "unet"): the earliest checkpoints (job
    1634 and its sibling seeds) were trained before ModelConfig gained an
    `architecture` field at all -- confirmed directly against the real
    checkpoint, not assumed -- back when this project only had one
    architecture, so every checkpoint missing the key was necessarily a
    plain U-Net. "unet" is exactly training/config.py's own current default
    for that field, so this isn't a guess bolted on here, it's applying the
    same default the dataclass would have applied if the field had existed
    at training time. Every checkpoint from job 1658 onward (U-Net++
    onward) carries the field explicitly -- checked directly too.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    model = build_model(
        architecture=model_cfg.get("architecture", "unet"),
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=None,
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["config"]


def _build_dataset_for_checkpoint(
    config: dict, normalization_stats: NormalizationStats, terrain_cache: TerrainCache
) -> Sen1Floods11TorchDataset:
    """
    Build the test dataset this specific checkpoint's own channel selection
    needs, reusing the shared terrain cache -- terrain doesn't depend on
    channel_indices (see this module's docstring), only which columns of
    the resulting tensor get kept.
    """
    dataset_cfg = config["dataset"]
    return Sen1Floods11TorchDataset(
        image_dir=DATA_ROOT / "S1Hand",
        label_dir=DATA_ROOT / "LabelHand",
        dem_dir=DATA_ROOT / "DEMHand",
        split_csv=DATA_ROOT / "splits" / TEST_SPLIT_CSV,
        normalization_stats=normalization_stats,
        terrain_cache=terrain_cache,
        augment=False,
        channel_indices=dataset_cfg.get("channel_indices"),
        jrc_dir=(DATA_ROOT / "JRCWaterHand") if dataset_cfg.get("include_jrc_baseline") else None,
        speckle_prob=0.0,
    )


def evaluate_checkpoint(
    entry: CheckpointEntry,
    device: str,
    normalization_stats: NormalizationStats,
    terrain_cache: TerrainCache,
) -> ScoredCheckpoint:
    start = time.time()
    checkpoint_path = entry.checkpoint_path()
    if not checkpoint_path.exists():
        return ScoredCheckpoint(
            entry, pooled=None, per_chip=None, seconds=0.0,
            error=f"checkpoint not found: {checkpoint_path}",
        )

    # Broad except deliberately: this is a batch sweep over checkpoints
    # trained across several sessions with slightly different config
    # schemas (see _load_model_for_eval's docstring on the missing
    # `architecture` key it already recovered from once). One checkpoint
    # hitting a config quirk this sweep hasn't seen yet must not crash the
    # whole run and lose the shared terrain cache -- expensive to rebuild
    # -- along with every other checkpoint queued behind it. Each failure
    # is recorded and printed, never silently swallowed.
    try:
        model, config = _load_model_for_eval(checkpoint_path, device)
        dataset = _build_dataset_for_checkpoint(config, normalization_stats, terrain_cache)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

        use_amp = device == "cuda"
        chip_metrics = predict_chip_metrics(
            model, dataloader, device, torch.bfloat16, use_amp, threshold=0.5, tta=entry.tta
        )
        pooled = aggregate_metrics(chip_metrics)
        per_chip = summarize(chip_metrics)
        return ScoredCheckpoint(entry, pooled, per_chip, seconds=time.time() - start)
    except Exception as exc:  # noqa: BLE001
        return ScoredCheckpoint(
            entry, pooled=None, per_chip=None, seconds=time.time() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def run() -> list[ScoredCheckpoint]:
    # "auto": actually detect CUDA rather than assuming it, so this degrades
    # to a (very slow) CPU smoke check instead of crashing if it's ever run
    # outside a real GPU allocation.
    device = resolve_device("auto")
    normalization_stats = NormalizationStats.load(NORMALIZATION_STATS_PATH)

    chip_ids = [
        image_filename.split("_S1Hand")[0]
        for image_filename, _ in read_split(DATA_ROOT / "splits" / TEST_SPLIT_CSV)
    ]
    print(f"building shared terrain cache for {len(chip_ids)} test chips...")
    terrain_start = time.time()
    terrain_cache = build_terrain_cache(chip_ids, DATA_ROOT / "DEMHand")
    print(f"  terrain cache built in {time.time() - terrain_start:.1f}s")

    results = []
    for entry in CHECKPOINTS:
        scored = evaluate_checkpoint(entry, device, normalization_stats, terrain_cache)
        results.append(scored)
        if scored.error:
            print(f"  [{entry.label}] SKIPPED: {scored.error}")
            continue
        p, c = scored.pooled, scored.per_chip
        print(
            f"  [{entry.label}] ({scored.seconds:.1f}s) pooled IoU {p.iou:.4f}, "
            f"per-chip IoU {c.mean_iou:.4f}, OA {p.overall_accuracy:.4f}, "
            f"kappa {p.kappa:.4f}"
        )
    return results


def print_summary_table(results: list[ScoredCheckpoint]) -> None:
    print(f"\n{'label':42} {'pooled':>8} {'perchip':>8} {'median':>8} {'OA':>8} {'kappa':>7}")
    for r in results:
        if r.error:
            print(f"{r.entry.label:42} SKIPPED ({r.error})")
            continue
        p, c = r.pooled, r.per_chip
        print(
            f"{r.entry.label:42} {p.iou:8.4f} {c.mean_iou:8.4f} {c.median_iou:8.4f} "
            f"{p.overall_accuracy:8.4f} {p.kappa:7.4f}"
        )


if __name__ == "__main__":
    scored_checkpoints = run()
    print_summary_table(scored_checkpoints)
