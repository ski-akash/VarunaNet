"""
One-off diagnostic: per-event IoU breakdown for a trained checkpoint on
the test split, to compare against benchmarks/RESULTS.md's per-event
classical-baseline table and see whether the CNN's aggregate mean_iou is
being dragged down by the same hard events (Ghana/USA/Pakistan) the
baselines already struggle with, or whether it's underperforming broadly.

Run: python -m training.per_event_eval checkpoint=<path>
"""

import hydra
import torch
from omegaconf import DictConfig

from benchmarks.metrics import compute_chip_metrics, summarize_per_event
from models.unet import build_unet
from training.checkpoint import load_checkpoint
from training.train import build_dataloader, resolve_device


def per_event_eval(cfg) -> dict:
    device = resolve_device(cfg.device)

    model = build_unet(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    ).to(device)
    load_checkpoint(cfg.checkpoint, model, map_location=device)
    model.eval()

    test_dataloader = build_dataloader(
        cfg, split_csv_name=cfg.dataset.test_split_csv_name, shuffle=False
    )

    chip_metrics = []
    with torch.no_grad():
        for inputs, targets, chip_ids in test_dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predicted_water = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy()
            targets_np = targets.numpy()
            for i, chip_id in enumerate(chip_ids):
                chip_metrics.append(
                    compute_chip_metrics(chip_id, predicted_water[i], targets_np[i])
                )

    return summarize_per_event(chip_metrics)


@hydra.main(config_path="conf", config_name="evaluate_test", version_base=None)
def main(cfg: DictConfig) -> None:
    per_event = per_event_eval(cfg)
    for event in sorted(per_event):
        s = per_event[event]
        print(f"{event:12s} mean_iou={s.mean_iou:.3f} median_iou={s.median_iou:.3f} n={s.n_chips}")


if __name__ == "__main__":
    main()
