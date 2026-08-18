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
    # "unet" or "unetplusplus" -- see models/unet.py's build_unet.
    architecture: str = "unet"
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
    # "synthetic" | "sen1floods11" -- see training/train.py's build_dataloader.
    name: str = "synthetic"
    batch_size: int = 2
    shuffle: bool = True

    # Only used when name == "synthetic".
    num_samples: int = 8
    height: int = 64
    width: int = 64

    # Only used when name == "sen1floods11". data_root matches the
    # on-disk layout VarunaNet_Spec.md's Environment notes describe:
    # <data_root>/{S1Hand,LabelHand,DEMHand,splits}/. normalization_stats_path
    # is the artifact data/compute_normalization_stats.py produces --
    # computed once on the training split and persisted, never
    # recomputed per-run (spec section 3.3).
    data_root: str = "datasets/sen1floods11"
    split_csv_name: str = "flood_train_data.csv"
    # Used for validation regardless of dataset name (see
    # training/train.py's build_val_dataloader) -- the official
    # Sen1Floods11 validation split, never touched for training itself.
    val_split_csv_name: str = "flood_valid_data.csv"
    # Used only by training/evaluate_test.py, for the final Phase 3 exit-
    # criterion comparison against the classical baselines -- never
    # touched during training or by validate_every_epochs' early-stopping
    # checks, both of which use val_split_csv_name instead. Scoring
    # against this split during training would leak it into every
    # training decision, defeating the point of holding it out.
    test_split_csv_name: str = "flood_test_data.csv"
    normalization_stats_path: str = "data/sen1floods11_normalization_stats.json"


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
    # Distinct from resume_from: loads only model weights from a prior
    # checkpoint (e.g. a weak-label pretraining run), leaving optimizer,
    # scheduler, global_step, and RNG state fresh -- for starting a new
    # fine-tuning run from pretrained weights, not resuming an
    # interrupted one.
    init_weights_from: Optional[str] = None
    # Spec section 4.2: "early stopping on val IoU". Validation runs every
    # validate_every_epochs epochs; training stops early if
    # early_stopping_patience validation checks in a row show no
    # improvement in mean val IoU.
    validate_every_epochs: int = 1
    early_stopping_patience: int = 5

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
