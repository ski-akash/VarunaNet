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

from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER, NUM_CHANNELS
from models.losses import build_loss
from models.unet import build_unet
from training.checkpoint import load_checkpoint as _load_checkpoint
from training.checkpoint import save_checkpoint, seed_everything
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
        return self.inputs[idx], self.labels[idx]


def build_dataloader(cfg) -> DataLoader:
    if cfg.dataset.name != "synthetic":
        raise NotImplementedError(
            f"dataset {cfg.dataset.name!r} isn't wired up yet -- only 'synthetic' is "
            "supported so far. Real Sen1Floods11 dataset wiring is a follow-up task."
        )
    dataset = SyntheticFloodDataset(
        num_samples=cfg.dataset.num_samples,
        height=cfg.dataset.height,
        width=cfg.dataset.width,
    )
    return DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)


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


def run_training(cfg, max_steps: int | None = None) -> dict:
    """
    Builds the model, loss, optimizer/scheduler, and dataloader from
    `cfg`; optionally resumes from cfg.resume_from; then trains for
    cfg.epochs worth of steps, checkpointing every
    cfg.checkpoint_every_steps steps (and always at the final step).

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
    steps_per_epoch = len(dataloader)
    total_steps = cfg.epochs * steps_per_epoch

    model = build_unet(
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    ).to(device)
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

    while global_step < stop_at:
        inputs, targets = next(data_iter)
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

    wandb.finish()
    return {"global_step": global_step, "final_loss": last_loss, "model": model}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(raw_cfg: DictConfig) -> None:
    # Merge the YAML-composed config against TrainConfig for type
    # validation -- see training/config.py's module docstring for why.
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), raw_cfg)
    result = run_training(cfg)
    print(f"training finished: step={result['global_step']} final_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
