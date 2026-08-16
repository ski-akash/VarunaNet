"""
Checkpoint save/resume and run-provenance utilities (spec section 8):
"Checkpoint every N steps, and resume automatically. SLURM jobs get
preempted and hit walltime limits; a run that cannot resume will cost
days" and "Deterministic seeding; log git commit SHA, config, and
environment with every run."

Resuming correctly means restoring more than just the model weights: the
optimizer's internal state (e.g. AdamW's running moment estimates), the LR
scheduler's position in its schedule, and the RNG state of every source of
randomness the loop touches (torch, numpy, python's random module -- see
seed_everything below) all have to come back too, or a "resumed" run
quietly continues from a different trajectory than an uninterrupted one
would have. That's worse than an obvious crash, per the same philosophy
data/contract.py applies to tensor shapes: a silent mismatch here would
produce results that differ from a true continuous run without ever
raising an error.

What deliberately is *not* restored: the exact position within the
dataloader's current epoch (only the completed-steps count is). Restoring
that exactly would mean checkpointing the sampler's internal state too,
which is disproportionate complexity for what it buys -- re-shuffling and
picking back up from the nearest step boundary loses at most one epoch's
worth of exact ordering, which does not matter for a shuffled training set.
"""

import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def get_git_commit_sha() -> str:
    """
    Returns the current commit SHA, or "unknown" if not run inside a git
    checkout (e.g. a stripped container image) -- logged into every
    checkpoint so a result can always be traced back to the exact code
    that produced it, without letting that requirement crash a run in an
    environment where git genuinely isn't available.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def seed_everything(seed: int) -> None:
    """
    Seeds every source of randomness this project's training loop touches,
    so a run is reproducible end to end. Only meaningful in combination
    with logging the seed (spec: "log the seed") and, on resume, restoring
    the RNG state saved in the checkpoint (see load_checkpoint) -- seeding
    once at the start of a run doesn't help if a resumed run silently
    starts back at the same initial RNG state instead of where it left off.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(rng_state: dict) -> None:
    torch.set_rng_state(rng_state["torch"])
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])


def save_checkpoint(
    path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    cfg,
    wandb_run_id: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg

    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config_dict,
            "git_commit_sha": get_git_commit_sha(),
            "rng_state": _capture_rng_state(),
            # Lets a resumed run log into the same logical W&B run instead
            # of starting a disconnected one -- see training/train.py's
            # run_training for how this gets passed back to wandb.init.
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def load_checkpoint(
    path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str = "cpu",
) -> dict:
    """
    Loads model/optimizer/scheduler state and restores RNG state in place,
    and returns the run-provenance metadata (global_step, epoch, the
    config the checkpoint was produced with, and its git commit SHA) so
    the caller can pick up training from exactly where it left off.

    weights_only=False: this checkpoint is our own trusted output (model
    state, optimizer state, RNG state, a plain config dict), never a file
    from an untrusted source, so the extra restrictions weights_only=True
    applies (meant for loading tensors from files you didn't produce
    yourself) aren't the right tradeoff here -- they'd reject the RNG
    state and config entries entirely.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    _restore_rng_state(checkpoint["rng_state"])

    return {
        "global_step": checkpoint["global_step"],
        "epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
        "git_commit_sha": checkpoint["git_commit_sha"],
        "wandb_run_id": checkpoint.get("wandb_run_id"),
    }


def find_latest_checkpoint(checkpoint_dir) -> Path | None:
    """
    Returns the highest-step checkpoint in checkpoint_dir (save_checkpoint
    names them step_<N>.pt), or None if the directory doesn't exist or has
    none yet.

    This is what makes "resume automatically" (spec section 8) actually
    automatic rather than something a human has to notice and wire up by
    hand: training/train.py's main() calls this before every run, so a
    SLURM job that gets killed and requeued (--requeue) just re-runs the
    exact same submitted command from scratch, and the training code
    itself finds and resumes from its own last checkpoint -- no special
    logic needed in the sbatch script, and no risk of a shell one-liner
    picking the wrong file via lexical instead of numeric sort (e.g.
    "step_10.pt" sorting before "step_2.pt").
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    checkpoints = []
    for path in checkpoint_dir.glob("step_*.pt"):
        try:
            step = int(path.stem.removeprefix("step_"))
        except ValueError:
            continue
        checkpoints.append((step, path))

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]
