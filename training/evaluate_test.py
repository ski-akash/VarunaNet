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

Run directly:
    python -m training.evaluate_test checkpoint=checkpoints/best.pt
    python -m training.evaluate_test checkpoint=checkpoints/best.pt tta=true
"""

import hydra
import torch
from omegaconf import DictConfig

from benchmarks.metrics import MetricSummary, compute_chip_metrics, summarize
from models.build_model import build_model
from training.checkpoint import load_checkpoint
from training.train import build_dataloader, resolve_device, run_validation


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


def evaluate_test(cfg) -> MetricSummary:
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

    if not getattr(cfg, "tta", False):
        return run_validation(
            model, test_dataloader, device, amp_dtype, use_amp, threshold=threshold
        )

    model.eval()
    chip_metrics = []
    with torch.no_grad():
        for inputs, targets, chip_ids in test_dataloader:
            inputs = inputs.to(device)
            probs = _tta_probabilities(model, inputs, amp_dtype, use_amp, device)
            predicted_water = (probs > threshold).squeeze(1).cpu().numpy()
            targets_np = targets.numpy()
            for i, chip_id in enumerate(chip_ids):
                chip_metrics.append(
                    compute_chip_metrics(chip_id, predicted_water[i], targets_np[i])
                )
    return summarize(chip_metrics)


@hydra.main(config_path="conf", config_name="evaluate_test", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = evaluate_test(cfg)
    tag = "tta" if getattr(cfg, "tta", False) else "no-tta"
    print(
        f"test split [{tag}] ({cfg.checkpoint}): "
        f"mean_iou={summary.mean_iou:.4f} median_iou={summary.median_iou:.4f} "
        f"mean_f1={summary.mean_f1:.4f} mean_precision={summary.mean_precision:.4f} "
        f"mean_recall={summary.mean_recall:.4f}"
    )


if __name__ == "__main__":
    main()
