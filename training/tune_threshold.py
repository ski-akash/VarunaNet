"""
Sweeps the sigmoid decision threshold on the *val* split to find the
value that maximizes mean IoU for a given checkpoint. Every other
evaluation in this project fixes the threshold at the standard 0.5
(models/losses.py's loss functions are implicitly built around it); this
is the one place that treats it as a tunable choice instead.

Deliberately scored only against val, never test -- matching this
project's existing train/val/test discipline (training/train.py's
run_validation already uses val for early stopping/best-checkpoint
selection, never test). Picking a threshold by looking at test scores
would leak the held-out split into a modeling decision, the same kind of
leak spec section 3.2 calls out for the official splits generally. Once
a threshold is chosen here, score it against test exactly once via
training/evaluate_test.py's own `threshold=` override.

Run directly:
    python -m training.tune_threshold checkpoint=checkpoints/best.pt
    python -m training.tune_threshold checkpoint=checkpoints/best.pt \
        thresholds=[0.3,0.4,0.5,0.6,0.7]
"""

import hydra
import torch
from omegaconf import DictConfig

from benchmarks.metrics import MetricSummary
from models.build_model import build_model
from training.checkpoint import load_checkpoint
from training.train import build_val_dataloader, resolve_device, run_validation


def tune_threshold(cfg) -> dict[float, MetricSummary]:
    device = resolve_device(cfg.device)
    model = build_model(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    ).to(device)
    load_checkpoint(cfg.checkpoint, model, map_location=device)

    val_dataloader = build_val_dataloader(cfg)
    use_amp = device == "cuda" and cfg.amp_dtype in ("fp16", "bf16")
    amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16

    results = {}
    for threshold in cfg.thresholds:
        results[threshold] = run_validation(
            model, val_dataloader, device, amp_dtype, use_amp, threshold=threshold
        )
    return results


@hydra.main(config_path="conf", config_name="tune_threshold", version_base=None)
def main(cfg: DictConfig) -> None:
    results = tune_threshold(cfg)
    best_threshold = max(results, key=lambda t: results[t].mean_iou)
    for threshold in sorted(results):
        marker = " <- best" if threshold == best_threshold else ""
        print(f"threshold={threshold:.2f} val_mean_iou={results[threshold].mean_iou:.4f}{marker}")
    print(f"best threshold: {best_threshold}")


if __name__ == "__main__":
    main()
