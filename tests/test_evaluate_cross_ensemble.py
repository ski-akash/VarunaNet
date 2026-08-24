"""
Tests for training/evaluate_cross_ensemble.py -- logit-averaging across
checkpoints from *different* architectures. Mirrors
tests/test_evaluate_ensemble.py's tiny synthetic-dataset approach: this
is about the multi-architecture loading-and-combining plumbing being
correct, not about model accuracy on real chips.
"""

from omegaconf import OmegaConf

from training.config import TrainConfig
from training.evaluate_cross_ensemble import evaluate_cross_ensemble
from training.train import run_training


def _tiny_cfg(checkpoint_dir, architecture="unet"):
    cfg = OmegaConf.structured(TrainConfig)
    cfg.model.architecture = architecture
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


def test_evaluate_cross_ensemble_scores_every_chip_across_architectures(tmp_path):
    # Train one tiny unet checkpoint and one tiny unetplusplus checkpoint
    # and confirm the cross-architecture ensemble path builds each with
    # its own architecture, loads both, and scores the full dataset --
    # the thing evaluate_ensemble.py's single shared `cfg.model` can't
    # express.
    unet_dir = tmp_path / "unet"
    unetpp_dir = tmp_path / "unetplusplus"
    run_training(_tiny_cfg(unet_dir, "unet"))
    run_training(_tiny_cfg(unetpp_dir, "unetplusplus"))

    cfg = OmegaConf.structured(TrainConfig)
    cfg.device = "cpu"
    cfg.dataset.num_samples = 4
    cfg.dataset.height = 64
    cfg.dataset.width = 64
    cfg.dataset.batch_size = 2
    OmegaConf.set_struct(cfg, False)
    cfg.amp_dtype = "fp32"
    cfg.members = [
        {"checkpoint": str(unet_dir / "best.pt"), "architecture": "unet"},
        {"checkpoint": str(unetpp_dir / "best.pt"), "architecture": "unetplusplus"},
    ]

    results = evaluate_cross_ensemble(cfg)

    assert results.per_chip.n_chips == cfg.dataset.num_samples
    assert results.pooled.n_chips == cfg.dataset.num_samples


def test_evaluate_cross_ensemble_tta_scores_every_chip(tmp_path):
    # tta=True switches member combination from logit- to
    # probability-averaging (see evaluate_cross_ensemble.py's
    # docstring) -- confirm that path also runs end to end and still
    # scores every chip, across two different architectures.
    unet_dir = tmp_path / "unet"
    unetpp_dir = tmp_path / "unetplusplus"
    run_training(_tiny_cfg(unet_dir, "unet"))
    run_training(_tiny_cfg(unetpp_dir, "unetplusplus"))

    cfg = OmegaConf.structured(TrainConfig)
    cfg.device = "cpu"
    cfg.dataset.num_samples = 4
    cfg.dataset.height = 64
    cfg.dataset.width = 64
    cfg.dataset.batch_size = 2
    OmegaConf.set_struct(cfg, False)
    cfg.amp_dtype = "fp32"
    cfg.tta = True
    cfg.members = [
        {"checkpoint": str(unet_dir / "best.pt"), "architecture": "unet"},
        {"checkpoint": str(unetpp_dir / "best.pt"), "architecture": "unetplusplus"},
    ]

    results = evaluate_cross_ensemble(cfg)

    assert results.per_chip.n_chips == cfg.dataset.num_samples
    assert results.pooled.n_chips == cfg.dataset.num_samples


def test_cross_ensemble_of_one_matches_single_checkpoint_evaluation(tmp_path):
    # Degenerate but meaningful check, mirroring
    # test_ensemble_of_one_matches_single_checkpoint_evaluation: a
    # cross-ensemble of exactly one member must reduce to the same
    # predictions evaluate_test.py's single-checkpoint path would
    # produce.
    from training.evaluate_test import evaluate_test

    cfg = _tiny_cfg(tmp_path, "unet")
    run_training(cfg)

    OmegaConf.set_struct(cfg, False)
    checkpoint_path = str(tmp_path / "best.pt")
    cfg.checkpoint = checkpoint_path
    single_results = evaluate_test(cfg)

    cfg.members = [{"checkpoint": checkpoint_path, "architecture": "unet"}]
    cross_results = evaluate_cross_ensemble(cfg)

    assert single_results.per_chip.mean_iou == cross_results.per_chip.mean_iou
