"""
Scores a trained checkpoint against the official Sen1Floods11 *test*
split -- the comparison Phase 3's exit criterion actually needs ("U-Net
beats all classical baselines", spec section 4.2), and something
training/train.py's run_validation() deliberately doesn't do itself: that
one scores the *val* split, for early stopping and best-checkpoint
selection during training. Scoring the test split there too would leak
it into every training decision, defeating the point of holding it out.
Test-split scoring only ever happens here, once, after training is done.

Uses the exact same benchmarks/metrics.py scoring the classical baselines
(Otsu, Otsu+HAND, Random Forest) were measured with in benchmarks/RESULTS.md,
so "beats baseline" is a real apples-to-apples comparison.

Reports BOTH the pooled/aggregate metrics (the number comparable to
Sen1Floods11's own published figures and the ~0.72 SOTA -- see spec section
15.1) and the per-chip mean (the stricter per-scene view this project has
always reported). Every classical and shallow-ML baseline in benchmarks/ was
re-scored to report both once the gap between them turned out to be large
(~0.17 pooled vs per-chip on Otsu); a CNN checkpoint scored only per-chip
here would not be comparable to any of them.

Run directly:
    python -m training.evaluate_test checkpoint=checkpoints/best.pt
    python -m training.evaluate_test checkpoint=checkpoints/best.pt tta=true
"""

from dataclasses import dataclass

import hydra
import torch
from omegaconf import DictConfig

from benchmarks.metrics import (
    AggregateMetrics,
    ChipMetrics,
    MetricSummary,
    aggregate_metrics,
    compute_chip_metrics,
    summarize,
)
from models.build_model import build_model
from training.checkpoint import load_checkpoint
from training.train import build_dataloader, resolve_device


@dataclass
class TestResults:
    pooled: AggregateMetrics
    per_chip: MetricSummary


def _tta_probabilities(model: torch.nn.Module, inputs: torch.Tensor, amp_dtype, use_amp, device):
    """
    Averages sigmoid probabilities across the horizontal flip, vertical
    flip, and both-flip variants of the input plus the original -- the
    same 4-way flip TTA a 2026 benchmark paper on this exact dataset
    reported a consistent +0.01-0.02 IoU gain from, across every
    architecture it tested (U-Net, U-Net++, DeepLabV3, SegFormer). Each
    flipped prediction is flipped back to the original orientation
    before averaging, so all four line up pixel-for-pixel.
    """
    flip_specs = [(), (-1,), (-2,), (-1, -2)]
    total = None
    for dims in flip_specs:
        flipped_input = torch.flip(inputs, dims=dims) if dims else inputs
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            logits = model(flipped_input)
        probs = torch.sigmoid(logits)
        probs = torch.flip(probs, dims=dims) if dims else probs
        total = probs if total is None else total + probs
    return total / len(flip_specs)


def _predict_chip_metrics(
    model: torch.nn.Module,
    dataloader,
    device: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
    threshold: float,
    tta: bool,
) -> list[ChipMetrics]:
    """
    Score every chip in `dataloader`, TTA or not, and return the raw per-chip
    metrics list. Deliberately not just calling training.train.run_validation
    (which only returns the already-summarized MetricSummary): computing both
    the pooled and per-chip views here needs the underlying ChipMetrics, with
    each one's ConfusionCounts intact, so aggregate_metrics can pool them.
    """
    model.eval()
    chip_metrics: list[ChipMetrics] = []
    with torch.no_grad():
        for inputs, targets, chip_ids in dataloader:
            inputs = inputs.to(device)
            if tta:
                probs = _tta_probabilities(model, inputs, amp_dtype, use_amp, device)
            else:
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    logits = model(inputs)
                probs = torch.sigmoid(logits)
            predicted_water = (probs > threshold).squeeze(1).cpu().numpy()
            targets_np = targets.numpy()
            for i, chip_id in enumerate(chip_ids):
                chip_metrics.append(
                    compute_chip_metrics(chip_id, predicted_water[i], targets_np[i])
                )
    return chip_metrics


def evaluate_test(cfg) -> TestResults:
    device = resolve_device(cfg.device)

    model = build_model(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    ).to(device)
    load_checkpoint(cfg.checkpoint, model, map_location=device)

    test_dataloader = build_dataloader(
        cfg, split_csv_name=cfg.dataset.test_split_csv_name, shuffle=False
    )

    use_amp = device == "cuda" and cfg.amp_dtype in ("fp16", "bf16")
    amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16
    threshold = getattr(cfg, "threshold", 0.5)
    tta = getattr(cfg, "tta", False)

    chip_metrics = _predict_chip_metrics(
        model, test_dataloader, device, amp_dtype, use_amp, threshold, tta
    )
    return TestResults(pooled=aggregate_metrics(chip_metrics), per_chip=summarize(chip_metrics))


@hydra.main(config_path="conf", config_name="evaluate_test", version_base=None)
def main(cfg: DictConfig) -> None:
    results = evaluate_test(cfg)
    tag = "tta" if getattr(cfg, "tta", False) else "no-tta"
    pooled, per_chip = results.pooled, results.per_chip
    print(
        f"test split [{tag}] ({cfg.checkpoint}):\n"
        f"  pooled:   IoU {pooled.iou:.4f}, F1 {pooled.f1:.4f}, "
        f"precision {pooled.precision:.4f}, recall {pooled.recall:.4f}, "
        f"OA {pooled.overall_accuracy:.4f}, kappa {pooled.kappa:.4f} "
        f"(n={pooled.n_chips} chips, {pooled.n_valid_pixels:,} valid px)\n"
        f"  per-chip: mean IoU {per_chip.mean_iou:.4f}, median IoU {per_chip.median_iou:.4f}, "
        f"mean F1 {per_chip.mean_f1:.4f}, mean precision {per_chip.mean_precision:.4f}, "
        f"mean recall {per_chip.mean_recall:.4f}"
    )


if __name__ == "__main__":
    main()
