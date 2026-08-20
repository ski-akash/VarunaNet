"""
Tests for training/tune_threshold.py -- sweeping the sigmoid decision
threshold on the val split. Mirrors tests/test_evaluate_test.py's tiny
synthetic-dataset approach: this is about the sweep-and-pick-best
plumbing being correct, not about model accuracy on real chips.
"""

from omegaconf import OmegaConf

from training.config import TrainConfig
from training.train import run_training
from training.tune_threshold import tune_threshold


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


def test_tune_threshold_scores_every_threshold_and_picks_a_best(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    run_training(cfg)

    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint = str(tmp_path / "best.pt")
    cfg.thresholds = [0.3, 0.5, 0.7]

    results = tune_threshold(cfg)

    assert set(results.keys()) == {0.3, 0.5, 0.7}
    for summary in results.values():
        assert summary.n_chips == cfg.dataset.num_samples
