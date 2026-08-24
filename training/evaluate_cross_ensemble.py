"""
Like evaluate_ensemble.py (logit-averaging across checkpoints), but each
checkpoint can be a *different* architecture -- e.g. one U-Net seed, one
U-Net++ checkpoint, one SegFormer-B2 checkpoint -- instead of several
seeds of the same one. evaluate_ensemble.py's `checkpoints` list assumes
a single shared `cfg.model`, which can't express "checkpoint A is a
unet, checkpoint B is a segformer_b2".

Motivated by this project's own cnn_results.md finding: same-architecture
logit-averaging (3 U-Net seeds) already beat every individual seed for
zero retraining cost. Diverse architectures making genuinely different
kinds of mistakes (CNN encoder-decoder vs. transformer) are a plausible
next step up from that -- same free-accuracy idea, applied across
architecture families instead of within one.

All member checkpoints must share the same input contract (5-channel
sen1floods11, 1 output logit -- true of every architecture in
training/conf/model/ as of this writing) so a single batch of inputs can
feed every model and their logits can be averaged directly.

`tta: true` (default false) folds evaluate_test.py's existing 4-way flip
TTA into each member's forward pass before combining -- reusing
_tta_probabilities rather than reimplementing it. With TTA on, each
member already returns flip-averaged sigmoid *probabilities* rather than
raw logits, so combining across members also happens in probability
space (not logit space) here; the two are different scales (this
project's earlier per-architecture TTA results show it isn't a universal
win, cnn_results.md), so this is worth comparing against the no-TTA
cross-ensemble rather than assumed to help.

Run directly, pointing `members` at a YAML list of
{checkpoint, architecture, encoder_name, encoder_weights, in_channels, classes}:
    python -m training.evaluate_cross_ensemble \
        members='[{checkpoint: a.pt, architecture: unet, encoder_name: resnet34}, ...]'
or, more practically, via a preset config (see conf/evaluate_cross_ensemble_top3.yaml).
"""

import hydra
import torch
from omegaconf import DictConfig

from benchmarks.metrics import ChipMetrics, aggregate_metrics, compute_chip_metrics, summarize
from models.build_model import build_model
from training.checkpoint import load_checkpoint
from training.evaluate_test import TestResults, _tta_probabilities
from training.train import build_dataloader, resolve_device


def _load_ensemble_models(members, device) -> list[torch.nn.Module]:
    models = []
    for member in members:
        model = build_model(
            architecture=member.architecture,
            encoder_name=member.get("encoder_name", "resnet34"),
            encoder_weights=None,  # a trained checkpoint loads next -- no imagenet fetch needed
            in_channels=member.get("in_channels", 5),
            classes=member.get("classes", 1),
        ).to(device)
        load_checkpoint(member.checkpoint, model, map_location=device)
        model.eval()
        models.append(model)
    return models


def evaluate_cross_ensemble(cfg) -> TestResults:
    device = resolve_device(cfg.device)
    models = _load_ensemble_models(cfg.members, device)
    use_tta = getattr(cfg, "tta", False)
    threshold = getattr(cfg, "threshold", 0.5)

    test_dataloader = build_dataloader(
        cfg, split_csv_name=cfg.dataset.test_split_csv_name, shuffle=False
    )

    use_amp = device == "cuda" and cfg.amp_dtype in ("fp16", "bf16")
    amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16

    chip_metrics: list[ChipMetrics] = []
    with torch.no_grad():
        for inputs, targets, chip_ids in test_dataloader:
            inputs = inputs.to(device)
            total = None
            for model in models:
                if use_tta:
                    member_output = _tta_probabilities(model, inputs, amp_dtype, use_amp, device)
                else:
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                        member_output = model(inputs)
                total = member_output if total is None else total + member_output
            averaged = total / len(models)
            probs = averaged if use_tta else torch.sigmoid(averaged)
            predicted_water = (probs > threshold).squeeze(1).cpu().numpy()
            targets_np = targets.numpy()
            for i, chip_id in enumerate(chip_ids):
                chip_metrics.append(
                    compute_chip_metrics(chip_id, predicted_water[i], targets_np[i])
                )
    return TestResults(pooled=aggregate_metrics(chip_metrics), per_chip=summarize(chip_metrics))


@hydra.main(config_path="conf", config_name="evaluate_cross_ensemble", version_base=None)
def main(cfg: DictConfig) -> None:
    results = evaluate_cross_ensemble(cfg)
    pooled, per_chip = results.pooled, results.per_chip
    architectures = ",".join(m.architecture for m in cfg.members)
    combine = "TTA probability-averaged" if getattr(cfg, "tta", False) else "logit-averaged"
    print(
        f"cross-architecture ensemble ({combine}, {len(cfg.members)} checkpoints: "
        f"{architectures}) test split:\n"
        f"  pooled:   IoU {pooled.iou:.4f}, F1 {pooled.f1:.4f}, "
        f"precision {pooled.precision:.4f}, recall {pooled.recall:.4f}, "
        f"OA {pooled.overall_accuracy:.4f}, kappa {pooled.kappa:.4f}\n"
        f"  per-chip: mean IoU {per_chip.mean_iou:.4f}, median IoU {per_chip.median_iou:.4f}, "
        f"mean F1 {per_chip.mean_f1:.4f}, mean precision {per_chip.mean_precision:.4f}, "
        f"mean recall {per_chip.mean_recall:.4f}"
    )


if __name__ == "__main__":
    main()
