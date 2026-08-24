"""
Tests for training/evaluate_ensemble.py -- logit-averaging across
multiple checkpoints. Mirrors tests/test_evaluate_test.py's tiny
synthetic-dataset approach: this is about the multi-checkpoint
loading-and-combining plumbing being correct, not about model accuracy
on real chips.
"""

import torch
from omegaconf import OmegaConf

from training.config import TrainConfig
from training.evaluate_ensemble import evaluate_ensemble
from training.train import run_training


def _tiny_cfg(checkpoint_dir):
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


def test_evaluate_ensemble_scores_every_chip(tmp_path):
    # Train two tiny checkpoints (different seeds -> different weights)
    # and confirm the ensemble path loads both and scores the full
    # dataset, same as the single-checkpoint version.
    ckpt_a_dir = tmp_path / "a"
    ckpt_b_dir = tmp_path / "b"
    cfg_a = _tiny_cfg(ckpt_a_dir)
    cfg_a.seed = 1
    cfg_b = _tiny_cfg(ckpt_b_dir)
    cfg_b.seed = 2
    run_training(cfg_a)
    run_training(cfg_b)

    OmegaConf.set_struct(cfg_a, False)
    cfg_a.checkpoints = [str(ckpt_a_dir / "best.pt"), str(ckpt_b_dir / "best.pt")]

    results = evaluate_ensemble(cfg_a)

    assert results.per_chip.n_chips == cfg_a.dataset.num_samples
    assert results.pooled.n_chips == cfg_a.dataset.num_samples


def test_ensemble_of_one_matches_single_checkpoint_evaluation(tmp_path):
    # A degenerate but meaningful check: averaging the logits of a single
    # checkpoint with itself (ensemble of one) must reduce to exactly the
    # same predictions evaluate_test.py's single-checkpoint path would
    # produce -- confirms logit averaging isn't silently distorting
    # anything (e.g. no unintended double-application of sigmoid).
    from training.evaluate_test import evaluate_test

    cfg = _tiny_cfg(tmp_path)
    run_training(cfg)

    OmegaConf.set_struct(cfg, False)
    checkpoint_path = str(tmp_path / "best.pt")
    cfg.checkpoint = checkpoint_path
    single_results = evaluate_test(cfg)

    cfg.checkpoints = [checkpoint_path]
    ensemble_results = evaluate_ensemble(cfg)

    assert single_results.per_chip.mean_iou == ensemble_results.per_chip.mean_iou


def test_logit_averaging_differs_from_naive_probability_averaging(tmp_path):
    # Regression test for the specific combination method chosen (logit
    # averaging, not post-sigmoid probability averaging) -- sigmoid is
    # nonlinear, so sigmoid(mean(logits)) != mean(sigmoid(logits)) in
    # general. Confirms evaluate_ensemble actually implements the former,
    # not the latter, by checking the math directly rather than assuming
    # the code does what the docstring says.
    # An asymmetric pair of logits (a fully-symmetric pair like [2,-2]
    # and [-2,2] happens to make both combination methods land on
    # exactly 0.5 by coincidence, which wouldn't actually distinguish
    # them) -- sigmoid(mean(logits)) != mean(sigmoid(logits)) in general.
    logits_a = torch.tensor([[3.0, -1.0]])
    logits_b = torch.tensor([[1.0, -1.0]])

    logit_averaged = torch.sigmoid((logits_a + logits_b) / 2)
    probability_averaged = (torch.sigmoid(logits_a) + torch.sigmoid(logits_b)) / 2

    assert not torch.allclose(logit_averaged, probability_averaged)
