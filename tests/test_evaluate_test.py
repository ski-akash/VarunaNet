"""
Tests for training/evaluate_test.py -- scoring a trained checkpoint
against the (held-out, per its own module docstring) test split.

Runs against the same tiny synthetic dataset training/train.py's own
tests use (see tests/test_train.py's _tiny_cfg) -- this is about the
checkpoint-load-then-score plumbing being correct, not about model
accuracy on real chips.
"""

from omegaconf import OmegaConf

from training.config import TrainConfig
from training.evaluate_test import evaluate_test
from training.train import run_training


def _tiny_cfg(checkpoint_dir):
    # Mirrors tests/test_train.py's own _tiny_cfg (not imported directly:
    # no other test file cross-imports from another here, and pytest's
    # no-__init__.py test layout makes that import path fragile).
    cfg = OmegaConf.structured(TrainConfig)
    cfg.model.encoder_weights = None
    cfg.device = "cpu"
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.dataset.num_samples = 4
    cfg.dataset.height = 64
    cfg.dataset.width = 64
    cfg.dataset.batch_size = 2
    cfg.epochs = 1
    cfg.checkpoint_every_steps = 1
    cfg.wandb.mode = "disabled"
    return cfg


def test_evaluate_test_scores_every_chip_from_a_saved_checkpoint(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    run_training(cfg)

    # cfg is OmegaConf.structured(TrainConfig) in struct mode, which
    # doesn't have a `checkpoint` field (deliberately -- see
    # training/conf/evaluate_test.yaml's own comment on why that field
    # lives in a separate config rather than TrainConfig). set_struct(False)
    # lets the test attach it dynamically rather than needing a second,
    # parallel cfg-building helper just for this one extra field.
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint = str(tmp_path / "best.pt")

    summary = evaluate_test(cfg)

    assert summary.n_chips == cfg.dataset.num_samples
