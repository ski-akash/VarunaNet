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
"""

import hydra
import torch
from omegaconf import DictConfig

from benchmarks.metrics import MetricSummary
from models.unet import build_unet
from training.checkpoint import load_checkpoint
from training.train import build_dataloader, resolve_device, run_validation


def evaluate_test(cfg) -> MetricSummary:
    device = resolve_device(cfg.device)

    model = build_unet(
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

    return run_validation(model, test_dataloader, device, amp_dtype, use_amp)


@hydra.main(config_path="conf", config_name="evaluate_test", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = evaluate_test(cfg)
    print(
        f"test split ({cfg.checkpoint}): "
        f"mean_iou={summary.mean_iou:.4f} median_iou={summary.median_iou:.4f} "
        f"mean_f1={summary.mean_f1:.4f} mean_precision={summary.mean_precision:.4f} "
        f"mean_recall={summary.mean_recall:.4f}"
    )


if __name__ == "__main__":
    main()
