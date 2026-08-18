"""
Config-driven training loop (spec section 8). Every path, hyperparameter,
and device assumption comes from `cfg` -- nothing here is hardcoded, so
the exact same code runs during local testing (tiny synthetic tensors, on
CPU, per section 8's "test the full training path locally on tiny
synthetic tensors before the user submits anything") and later on the
SLURM cluster (real Sen1Floods11 chips, GPU) with only the config
changed.

`run_training` is a plain function, deliberately kept separate from the
`@hydra.main`-decorated `main()` entry point below, so tests can build a
config directly and call it without going through Hydra's CLI argument
parsing.
"""

import math
from collections.abc import Iterator
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from benchmarks.metrics import MetricSummary, compute_chip_metrics, summarize
from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER, NUM_CHANNELS
from models.losses import build_loss
from models.unet import build_unet
from training.checkpoint import find_latest_checkpoint, save_checkpoint, seed_everything
from training.checkpoint import load_checkpoint as _load_checkpoint
from training.config import TrainConfig


def _infinite_batches(dataloader: DataLoader) -> Iterator:
    """
    Yields batches forever, re-iterating the dataloader once each pass is
    exhausted. Deliberately not itertools.cycle(dataloader): cycle() only
    calls iter(dataloader) on the very first pass, then caches every
    yielded batch and replays that same cached sequence on every
    subsequent lap -- it never calls iter(dataloader) again. Since
    DataLoader's shuffling happens inside iter(dataloader) (a fresh
    RandomSampler permutation each call), that would silently mean every
    epoch after the first replays the exact same batch order forever,
    defeating shuffling for nearly the whole run. Confirmed directly
    (itertools.cycle([1,2,3,4]) for 8 draws returns [1,2,3,4,1,2,3,4], the
    identical order twice) before writing this replacement.
    """
    while True:
        yield from dataloader


def resolve_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


class SyntheticFloodDataset(Dataset):
    """
    Fixed, seeded synthetic tensors matching the data contract's shapes
    (data/contract.py) at a configurable tiny size, so the training loop
    can be exercised end to end -- forward pass, loss, backward pass,
    optimizer step, checkpointing -- without needing real imagery.

    Generated once at construction time from its own generator, seeded
    independently of the training run's seed_everything() call -- the
    *data* a run trains on should stay identical across runs regardless of
    training seed; only things like shuffle order and weight
    initialization are meant to vary with the training seed.
    """

    def __init__(self, num_samples: int, height: int, width: int, seed: int = 12345):
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(num_samples, NUM_CHANNELS, height, width, generator=generator)
        self.labels = torch.randint(
            LABEL_NON_WATER,
            LABEL_WATER + 1,
            (num_samples, height, width),
            generator=generator,
        ).long()
        # Simulate a strip of no-data on one sample, like a real
        # scene-edge chip (see VarunaNet_Spec.md's known real-data
        # quirks), so the loop actually exercises the ignore-index path
        # rather than only the easy all-valid case.
        if num_samples > 0:
            self.labels[0, :4, :] = LABEL_IGNORE

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        # A synthetic-but-real-shaped chip id (not just the input/label
        # tensors), so run_validation exercises the exact same code path
        # against synthetic data that it does against real chips --
        # benchmarks.metrics.compute_chip_metrics needs a chip id for
        # every prediction it scores, real or synthetic.
        return self.inputs[idx], self.labels[idx], f"synthetic_{idx}"


def build_dataloader(
    cfg,
    split_csv_name: str | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """
    split_csv_name/shuffle let build_val_dataloader (below) reuse this
    function for the validation split instead of duplicating the
    dataset-construction branch -- validation always wants a different
    split file and shuffle=False (order doesn't matter for scoring, and
    a fixed order makes debugging a specific chip's score easier).
    """
    if cfg.dataset.name == "synthetic":
        dataset = SyntheticFloodDataset(
            num_samples=cfg.dataset.num_samples,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
        )
    elif cfg.dataset.name == "sen1floods11":
        # Local imports: keeps torch's synthetic-only test path from
        # needing rasterio/pysheds (real chip I/O and terrain
        # computation) just to import this module at all.
        from data.normalization import NormalizationStats
        from training.sen1floods11_dataset import build_sen1floods11_dataset

        stats = NormalizationStats.load(cfg.dataset.normalization_stats_path)
        dataset = build_sen1floods11_dataset(
            data_root=cfg.dataset.data_root,
            split_csv_name=(
                split_csv_name if split_csv_name is not None else cfg.dataset.split_csv_name
            ),
            normalization_stats=stats,
            # Only the training split gets flip augmentation -- validation
            # and test (called with an explicit split_csv_name) must stay
            # deterministic so scores are comparable run to run.
            augment=split_csv_name is None,
        )
    elif cfg.dataset.name == "sen1floods11_weak":
        # Weak-label pretraining set (data/sen1floods11_weak.py) -- no
        # official split, so "train"/"val" here means the small held-out
        # slice split_weak_chip_ids carves off, not a benchmark split.
        # split_csv_name is unused for this dataset (kept as the shared
        # build_dataloader parameter name for build_val_dataloader below).
        from data.normalization import NormalizationStats
        from training.sen1floods11_weak_dataset import build_sen1floods11_weak_dataset

        stats = NormalizationStats.load(cfg.dataset.normalization_stats_path)
        dataset = build_sen1floods11_weak_dataset(
            data_root=cfg.dataset.data_root,
            normalization_stats=stats,
            split="train" if split_csv_name is None else "val",
            augment=split_csv_name is None,
        )
    else:
        raise NotImplementedError(
            f"dataset {cfg.dataset.name!r} isn't wired up yet -- 'synthetic' and "
            "'sen1floods11' are the only options implemented so far."
        )
    resolved_shuffle = cfg.dataset.shuffle if shuffle is None else shuffle
    return DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=resolved_shuffle)


def build_val_dataloader(cfg) -> DataLoader:
    return build_dataloader(cfg, split_csv_name=cfg.dataset.val_split_csv_name, shuffle=False)


def _loss_kwargs(loss_cfg) -> dict:
    """Only the fields the selected loss actually accepts -- see build_loss."""
    if loss_cfg.name == "dice_bce":
        return {"dice_weight": loss_cfg.dice_weight, "bce_weight": loss_cfg.bce_weight}
    if loss_cfg.name == "focal":
        return {"gamma": loss_cfg.gamma, "alpha": loss_cfg.alpha}
    return {}


def build_optimizer_and_scheduler(cfg, model: torch.nn.Module, total_steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
    )

    warmup_steps = cfg.scheduler.warmup_steps
    min_lr_ratio = cfg.scheduler.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def run_validation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> MetricSummary:
    """
    Scores the model against `dataloader` with benchmarks/metrics.py --
    the exact same per-chip IoU/F1/precision/recall computation the
    classical baselines (Otsu, Otsu+HAND, Random Forest) were measured
    with in benchmarks/RESULTS.md. That's not incidental: Phase 3's exit
    criterion is "U-Net beats all classical baselines", which is only a
    meaningful comparison if both sides were scored the same way, not by
    two different metric implementations that might disagree at the
    margins.

    Threshold is a fixed 0.5 on sigmoid(logits) -- the standard default
    for binary segmentation, and the same implicit threshold the loss
    functions (models/losses.py) are built around.
    """
    model.eval()
    chip_metrics = []
    with torch.no_grad():
        for inputs, targets, chip_ids in dataloader:
            inputs = inputs.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(inputs)
            predicted_water = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy()
            targets_np = targets.numpy()
            for i, chip_id in enumerate(chip_ids):
                chip_metrics.append(
                    compute_chip_metrics(chip_id, predicted_water[i], targets_np[i])
                )
    model.train()
    return summarize(chip_metrics)


def run_training(cfg, max_steps: int | None = None) -> dict:
    """
    Builds the model, loss, optimizer/scheduler, and dataloader from
    `cfg`; optionally resumes from cfg.resume_from; then trains for
    cfg.epochs worth of steps, checkpointing every
    cfg.checkpoint_every_steps steps (and always at the final step).
    Validates against the held-out split every cfg.validate_every_epochs
    epochs, saving a separate best.pt checkpoint whenever val IoU
    improves, and stops early if it hasn't improved for
    cfg.early_stopping_patience validation checks in a row (spec section
    4.2: "early stopping on val IoU").

    cfg.epochs always determines the *overall* training target -- it must
    stay the same across every invocation of a given run, resumed or not,
    because it also determines the LR scheduler's total_steps (the length
    of the cosine decay). A real SLURM preemption stops the process
    externally (SIGTERM/walltime) without the loop's own knowledge, so
    `max_steps` exists purely so callers (tests, mainly) can simulate that
    same kind of early stop deliberately, without conflating it with
    shrinking the actual training target -- doing the latter would give
    the interrupted run a different (shorter) LR schedule than an
    uninterrupted run configured with the same cfg.epochs would have used,
    silently producing different results for a reason unrelated to
    resuming itself.

    Returns a small summary dict so callers/tests can assert on the
    outcome without re-deriving it from checkpoint files.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    dataloader = build_dataloader(cfg)
    val_dataloader = build_val_dataloader(cfg)
    steps_per_epoch = len(dataloader)
    total_steps = cfg.epochs * steps_per_epoch

    model = build_unet(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    ).to(device)

    if cfg.init_weights_from is not None:
        # weights_only=False: same trusted-checkpoint reasoning as
        # load_checkpoint below -- this file is our own prior run's
        # output, not an untrusted source.
        pretrained = torch.load(cfg.init_weights_from, map_location=device, weights_only=False)
        model.load_state_dict(pretrained["model_state_dict"])

    loss_fn = build_loss(cfg.loss.name, **_loss_kwargs(cfg.loss))
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, total_steps)

    global_step = 0
    wandb_run_id = None
    if cfg.resume_from is not None:
        meta = _load_checkpoint(cfg.resume_from, model, optimizer, scheduler, map_location=device)
        global_step = meta["global_step"]
        wandb_run_id = meta["wandb_run_id"]

    # id + resume="allow" lets a resumed run log into the same logical
    # W&B run as the one that got interrupted, rather than starting a
    # disconnected new one, whenever wandb_run_id was recovered from a
    # checkpoint above. Note (verified directly, not assumed from docs):
    # under mode="offline" this does NOT append to the original run's
    # local files -- W&B prints "resume will be ignored... starting a new
    # run with run id X" and writes a second local run directory, but it
    # does keep the same run id. That's expected offline-mode behavior,
    # not a bug here: syncing both local directories later (`wandb sync`)
    # reconciles them into one run on the dashboard because they share an
    # id, which is the whole reason wandb_run_id is threaded through the
    # checkpoint at all.
    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        dir=cfg.wandb.dir,
        config=OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else None,
        id=wandb_run_id,
        resume="allow" if wandb_run_id is not None else None,
    )

    # AMP only actually activates on CUDA -- see models section 4.2's note
    # that bf16/fp16 are a GPU concern (A100 vs V100). On CPU, running
    # fp32 regardless of amp_dtype keeps local/dev testing simple and
    # correct rather than chasing CPU-autocast edge cases that don't
    # matter for where AMP is actually meant to pay off.
    use_amp = device == "cuda" and cfg.amp_dtype in ("fp16", "bf16")
    amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16
    use_scaler = use_amp and cfg.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    model.train()
    data_iter = _infinite_batches(dataloader)
    last_loss = None
    stop_at = total_steps if max_steps is None else min(total_steps, global_step + max_steps)
    best_val_iou = float("-inf")
    epochs_without_improvement = 0
    stopped_early = False

    while global_step < stop_at:
        inputs, targets, _ = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            logits = model(inputs)
            loss = loss_fn(logits, targets)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        global_step += 1
        last_loss = loss.item()

        if global_step % cfg.wandb.log_every_steps == 0:
            wandb.log(
                {"train/loss": last_loss, "train/lr": scheduler.get_last_lr()[0]},
                step=global_step,
            )

        # Validation runs *before* the periodic checkpoint below,
        # deliberately: iterating a DataLoader consumes one draw from the
        # global RNG even with shuffle=False (it generates a base seed for
        # potential worker processes regardless of num_workers -- confirmed
        # directly, not assumed, since it's a genuinely non-obvious PyTorch
        # internal). If the periodic checkpoint ran first, it would capture
        # RNG state from *before* validation's draw, so a resumed run
        # wouldn't replay that draw the way an uninterrupted run's
        # continuous RNG stream naturally would -- the two would silently
        # diverge at the next shuffle. Ordering validation first means the
        # checkpoint always captures the true post-validation state.
        if global_step % steps_per_epoch == 0:
            epoch = global_step // steps_per_epoch
            if epoch % cfg.validate_every_epochs == 0:
                val_summary = run_validation(model, val_dataloader, device, amp_dtype, use_amp)
                print(
                    f"epoch {epoch} step {global_step} loss={last_loss:.4f} "
                    f"val_mean_iou={val_summary.mean_iou:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                wandb.log(
                    {
                        "val/mean_iou": val_summary.mean_iou,
                        "val/median_iou": val_summary.median_iou,
                        "val/mean_f1": val_summary.mean_f1,
                        "val/mean_precision": val_summary.mean_precision,
                        "val/mean_recall": val_summary.mean_recall,
                    },
                    step=global_step,
                )

                if val_summary.mean_iou > best_val_iou:
                    best_val_iou = val_summary.mean_iou
                    epochs_without_improvement = 0
                    save_checkpoint(
                        Path(cfg.checkpoint_dir) / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        cfg,
                        wandb_run_id=wandb_run.id,
                    )
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= cfg.early_stopping_patience:
                        stopped_early = True

        if global_step % cfg.checkpoint_every_steps == 0 or global_step == total_steps:
            epoch = global_step // steps_per_epoch
            checkpoint_path = Path(cfg.checkpoint_dir) / f"step_{global_step}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                global_step,
                epoch,
                cfg,
                wandb_run_id=wandb_run.id,
            )

        if stopped_early:
            break

    wandb.finish()
    return {
        "global_step": global_step,
        "final_loss": last_loss,
        "model": model,
        "best_val_iou": best_val_iou if best_val_iou != float("-inf") else None,
        "stopped_early": stopped_early,
    }


def _maybe_resume_from_latest(cfg) -> None:
    """
    "Resume automatically" (spec section 8): if the config didn't already
    say where to resume from, check whether this run's own checkpoint_dir
    already has one and use it. This is what a requeued SLURM job
    (--requeue, see training/submit_train.sbatch) relies on -- it re-runs
    the exact same submitted command from scratch, and this is the only
    place that needs to notice a checkpoint already exists; nothing
    special is needed in the sbatch script itself.

    Kept separate from run_training() itself so that function always just
    does exactly what cfg.resume_from says -- tests that call it directly
    stay predictable and aren't surprised by filesystem state -- and
    separate from main() so this auto-detection step is testable without
    going through Hydra's CLI-parsing decorator.
    """
    if cfg.resume_from is not None:
        return
    latest = find_latest_checkpoint(cfg.checkpoint_dir)
    if latest is not None:
        print(f"found existing checkpoint, resuming from: {latest}")
        cfg.resume_from = str(latest)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    # Merge the YAML-composed config against TrainConfig for type
    # validation -- see training/config.py's module docstring for why.
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), raw_cfg)
    _maybe_resume_from_latest(cfg)
    result = run_training(cfg)
    print(f"training finished: step={result['global_step']} final_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
