"""
Structured configs for the training loop (spec section 8: "config-driven
-- Hydra or YAML. Zero hardcoded paths, hyperparameters, or device
assumptions"). These are plain dataclasses merged against the YAML files
in training/conf/ at load time (see training/train.py's build_config),
rather than raw YAML loaded straight into untyped dicts -- a typo'd or
wrong-typed field in a YAML config then fails immediately and loudly at
load time, instead of surfacing hours later as a confusing crash (or
worse, a silently wrong value) deep inside the training loop. That's the
same philosophy data/contract.py applies to tensor shapes, applied here to
configuration.
"""

from dataclasses import dataclass, field
from typing import Optional

from data.contract import NUM_CHANNELS


@dataclass
class ModelConfig:
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = NUM_CHANNELS
    classes: int = 1


@dataclass
class LossConfig:
    # "dice_bce" or "focal" -- see models/losses.py's build_loss. Not
    # every field below applies to every name; training/train.py picks
    # only the ones the selected loss actually accepts.
    name: str = "dice_bce"
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    gamma: float = 2.0
    alpha: Optional[float] = 0.25


@dataclass
class OptimizerConfig:
    # AdamW, per spec section 4.2.
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    # Cosine schedule with warmup, per spec section 4.2. min_lr_ratio is
    # the fraction of the base LR the cosine curve decays down to by the
    # end of training, rather than decaying all the way to zero.
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0


@dataclass
class DatasetConfig:
    # "synthetic" is the only dataset wired up so far -- real Sen1Floods11
    # dataset wiring is a follow-up task (see VarunaNet_Spec.md).
    name: str = "synthetic"
    num_samples: int = 8
    height: int = 64
    width: int = 64
    batch_size: int = 2
    shuffle: bool = True


@dataclass
class WandbConfig:
    project: str = "varunanet"
    entity: Optional[str] = None
    # "offline" by default -- spec section 8: "Weights & Biases logging
    # (offline mode with later sync, since compute nodes are often
    # air-gapped)". Local run data still gets written either way; offline
    # just skips live network calls during training, synced afterward via
    # `wandb sync`.
    mode: str = "offline"
    log_every_steps: int = 1
    # None -> wandb's own default (./wandb/ under the current working
    # directory). Overridden in tests to keep run files out of the repo.
    dir: Optional[str] = None


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    # Spec section 4.2: "bf16 requires A100 (Ampere); on V100 use fp16
    # with a gradient scaler. Make the dtype a config field, do not
    # hardcode." -- "fp32" | "fp16" | "bf16".
    amp_dtype: str = "fp32"
    epochs: int = 1
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_steps: int = 50
    resume_from: Optional[str] = None

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
