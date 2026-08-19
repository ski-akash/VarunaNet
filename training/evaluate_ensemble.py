"""
Averages predictions across multiple checkpoints of the same
architecture, scored against the official Sen1Floods11 test split -- a
classic, zero-retrain way to reduce variance and often beat every
individual seed, on top of the 3-seed mean +/- std this project already
reports (see training/evaluate_test.py, the single-checkpoint version
this mirrors).

Combines checkpoints by averaging their raw logits, then applying a
single sigmoid -- not by averaging each model's own post-sigmoid
probability first. The two are mathematically different (sigmoid is
nonlinear), and logit averaging is the simpler, more direct combination:
it doesn't compress each model's confidence into [0, 1] before
combining, so a model that's very confident isn't discounted by the
squashing the other approach would apply first.

Run directly:
    python -m training.evaluate_ensemble checkpoints=[a.pt,b.pt,c.pt]
"""

import hydra
import torch
from omegaconf import DictConfig

from benchmarks.metrics import MetricSummary, compute_chip_metrics, summarize
from models.build_model import build_model
from training.checkpoint import load_checkpoint
from training.train import build_dataloader, resolve_device


def _load_ensemble_models(checkpoints, cfg, device) -> list[torch.nn.Module]:
    models = []
    for checkpoint_path in checkpoints:
        model = build_model(
            architecture=cfg.model.architecture,
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.model.classes,
        ).to(device)
        load_checkpoint(checkpoint_path, model, map_location=device)
        model.eval()
        models.append(model)
    return models


def evaluate_ensemble(cfg) -> MetricSummary:
    device = resolve_device(cfg.device)
    models = _load_ensemble_models(cfg.checkpoints, cfg, device)

    test_dataloader = build_dataloader(
        cfg, split_csv_name=cfg.dataset.test_split_csv_name, shuffle=False
    )

    use_amp = device == "cuda" and cfg.amp_dtype in ("fp16", "bf16")
    amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16

    chip_metrics = []
    with torch.no_grad():
        for inputs, targets, chip_ids in test_dataloader:
            inputs = inputs.to(device)
            total_logits = None
            for model in models:
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    logits = model(inputs)
                total_logits = logits if total_logits is None else total_logits + logits
            averaged_logits = total_logits / len(models)
            predicted_water = (torch.sigmoid(averaged_logits) > 0.5).squeeze(1).cpu().numpy()
            targets_np = targets.numpy()
            for i, chip_id in enumerate(chip_ids):
                chip_metrics.append(
                    compute_chip_metrics(chip_id, predicted_water[i], targets_np[i])
                )
    return summarize(chip_metrics)


@hydra.main(config_path="conf", config_name="evaluate_ensemble", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = evaluate_ensemble(cfg)
    print(
        f"ensemble (logit-averaged, {len(cfg.checkpoints)} checkpoints) test split: "
        f"mean_iou={summary.mean_iou:.4f} median_iou={summary.median_iou:.4f} "
        f"mean_f1={summary.mean_f1:.4f} mean_precision={summary.mean_precision:.4f} "
        f"mean_recall={summary.mean_recall:.4f}"
    )


if __name__ == "__main__":
    main()
