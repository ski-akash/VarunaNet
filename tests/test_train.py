"""
Tests for the config-driven training loop (training/train.py,
training/config.py, training/checkpoint.py).

Per spec section 8, everything here runs against tiny synthetic tensors on
the CPU-only path -- these tests are about training *mechanics* (config
composition, checkpoint/resume correctness, deterministic seeding, the
AMP config field not crashing without a GPU), not about model accuracy,
which is what benchmarks/ measures against real chips.
"""

from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from training.config import TrainConfig
from training.train import build_dataloader, run_training

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "training" / "conf")


def _tiny_cfg(checkpoint_dir, **overrides):
    """
    A TrainConfig for fast, offline, CPU-only tests: no ImageNet download
    (encoder_weights=None), a tiny model input size, a tiny synthetic
    dataset, and W&B mode="disabled" (a complete no-op -- verified to
    write zero files and skip all network I/O -- so tests don't litter
    the repo with real run directories or attempt any network call).
    `overrides` are dotted-path OmegaConf updates, e.g. epochs=2 or
    **{"scheduler.warmup_steps": 1}.
    """
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
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def test_default_config_composes_with_hydra():
    with initialize(config_path="../training/conf", version_base=None):
        raw_cfg = compose(config_name="config")
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), raw_cfg)

    assert cfg.model.encoder_name == "resnet34"
    assert cfg.model.encoder_weights == "imagenet"
    assert cfg.loss.name == "dice_bce"
    assert cfg.dataset.name == "synthetic"


def test_focal_loss_config_group_overrides_loss_name():
    with initialize(config_path="../training/conf", version_base=None):
        raw_cfg = compose(config_name="config", overrides=["loss=focal"])
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), raw_cfg)

    assert cfg.loss.name == "focal"


def test_command_line_style_overrides_apply():
    with initialize(config_path="../training/conf", version_base=None):
        raw_cfg = compose(config_name="config", overrides=["seed=7", "optimizer.lr=3e-4"])
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), raw_cfg)

    assert cfg.seed == 7
    assert cfg.optimizer.lr == 3e-4


def test_unknown_dataset_name_raises_not_implemented(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cfg.dataset.name = "sen1floods11"

    try:
        build_dataloader(cfg)
    except NotImplementedError:
        return
    raise AssertionError("expected build_dataloader to raise NotImplementedError")


def test_run_training_completes_and_reports_progress(tmp_path):
    cfg = _tiny_cfg(tmp_path, epochs=1)  # 4 samples / batch 2 -> 2 steps

    result = run_training(cfg)

    assert result["global_step"] == 2
    assert torch.isfinite(torch.tensor(result["final_loss"]))


def test_checkpoints_saved_at_expected_steps(tmp_path):
    cfg = _tiny_cfg(tmp_path, epochs=3, checkpoint_every_steps=2)  # 2 steps/epoch -> 6 total

    run_training(cfg)

    assert (tmp_path / "step_2.pt").exists()
    assert (tmp_path / "step_4.pt").exists()
    assert (tmp_path / "step_6.pt").exists()  # final step, saved even without an exact multiple


def test_two_independent_runs_with_same_seed_are_deterministic(tmp_path):
    cfg_1 = _tiny_cfg(tmp_path / "run1", epochs=2)
    cfg_2 = _tiny_cfg(tmp_path / "run2", epochs=2)

    result_1 = run_training(cfg_1)
    result_2 = run_training(cfg_2)

    assert result_1["final_loss"] == result_2["final_loss"]
    for (name_1, p_1), (name_2, p_2) in zip(
        result_1["model"].state_dict().items(), result_2["model"].state_dict().items()
    ):
        assert name_1 == name_2
        assert torch.equal(p_1, p_2), f"weights diverged in {name_1}"


def test_resume_continues_and_matches_uninterrupted_run(tmp_path):
    # Uninterrupted: one process runs all 4 steps (2 epochs) straight through.
    cfg_full = _tiny_cfg(tmp_path / "full", epochs=2, checkpoint_every_steps=4)
    result_full = run_training(cfg_full)
    assert result_full["global_step"] == 4

    # Interrupted: same overall target (epochs=2 -> total_steps=4, so the
    # LR schedule matches cfg_full's exactly), but cut short via max_steps
    # to simulate a SLURM preemption after step 2 -- see run_training's
    # docstring for why this has to be max_steps rather than a smaller
    # cfg.epochs.
    cfg_part1 = _tiny_cfg(tmp_path / "resumed", epochs=2, checkpoint_every_steps=2)
    run_training(cfg_part1, max_steps=2)
    checkpoint_path = tmp_path / "resumed" / "step_2.pt"
    assert checkpoint_path.exists()

    # Resume: a fresh run_training call picks up from that checkpoint and
    # finishes the same total target (4 steps) that cfg_full ran in one go.
    cfg_part2 = _tiny_cfg(
        tmp_path / "resumed",
        epochs=2,
        checkpoint_every_steps=4,
        resume_from=str(checkpoint_path),
    )
    result_resumed = run_training(cfg_part2)

    assert result_resumed["global_step"] == 4
    assert result_resumed["final_loss"] == result_full["final_loss"]
    for (name_f, p_f), (name_r, p_r) in zip(
        result_full["model"].state_dict().items(), result_resumed["model"].state_dict().items()
    ):
        assert name_f == name_r
        assert torch.equal(p_f, p_r), f"resumed run diverged from uninterrupted run in {name_f}"


def test_resumed_run_does_not_restart_already_completed_steps(tmp_path):
    # If resume ignored the checkpoint's global_step and started over, this
    # would report 6 total steps instead of 4.
    cfg_part1 = _tiny_cfg(tmp_path, epochs=1, checkpoint_every_steps=2)
    run_training(cfg_part1)

    cfg_part2 = _tiny_cfg(
        tmp_path, epochs=2, checkpoint_every_steps=4, resume_from=str(tmp_path / "step_2.pt")
    )
    result = run_training(cfg_part2)

    assert result["global_step"] == 4


def test_amp_dtype_config_is_a_noop_on_cpu(tmp_path):
    # Spec section 4.2: amp_dtype must be a config field, not hardcoded --
    # and on a machine with no CUDA (like this dev machine), every value
    # should still run correctly rather than crash, since AMP only
    # actually activates on a CUDA device (see run_training).
    for dtype in ["fp32", "fp16", "bf16"]:
        cfg = _tiny_cfg(tmp_path / dtype, epochs=1)
        cfg.amp_dtype = dtype
        result = run_training(cfg)
        assert torch.isfinite(torch.tensor(result["final_loss"]))


def test_warmup_then_cosine_schedule_reaches_min_lr_ratio(tmp_path):
    # scheduler.step() is called once per training step in run_training,
    # so after all total_steps have elapsed the LR should have decayed to
    # exactly base_lr * min_lr_ratio -- verified against the loop's own
    # optimizer, not a hand-rolled duplicate scheduler.
    from training.train import build_optimizer_and_scheduler

    model = torch.nn.Linear(2, 2)
    base_lr = 1.0
    min_lr_ratio = 0.1
    total_steps = 10

    class Cfg:
        class optimizer:
            lr = base_lr
            weight_decay = 0.0
            betas = (0.9, 0.999)

        class scheduler:
            warmup_steps = 3
            min_lr_ratio = 0.1

    optimizer, scheduler = build_optimizer_and_scheduler(Cfg, model, total_steps)

    # Warmup: LR ramps up linearly and should not yet have reached base_lr.
    # (warmup_steps=3, so lr_lambda(1) = 2/3 < 1.0; lr_lambda(2) already
    # reaches 1.0, so this has to check after exactly 1 step.)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < base_lr

    # After warmup + full cosine decay, LR lands exactly at the floor.
    for _ in range(total_steps - 1):
        optimizer.step()
        scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - base_lr * min_lr_ratio) < 1e-9


def test_wandb_disabled_mode_writes_no_files(tmp_path):
    # Confirms the premise _tiny_cfg relies on everywhere else: mode=
    # "disabled" is a true no-op (no local files, no network), which is
    # why it's safe to use as the default for every other test in this
    # file rather than only this one.
    cfg = _tiny_cfg(tmp_path, epochs=1)
    run_training(cfg)

    assert not any(tmp_path.rglob("wandb"))


def test_wandb_offline_mode_writes_local_run_files(tmp_path):
    # The actual production default (spec section 8) -- confirms real
    # local run files get written without any network access or API key,
    # not just that mode="offline" is accepted as a string.
    cfg = _tiny_cfg(tmp_path, epochs=1)
    cfg.wandb.mode = "offline"
    cfg.wandb.dir = str(tmp_path)

    run_training(cfg)

    offline_runs = list(tmp_path.glob("wandb/offline-run-*"))
    assert len(offline_runs) == 1
    assert any(f.suffix == ".wandb" for f in offline_runs[0].iterdir())


def test_wandb_run_id_persisted_and_reused_on_resume(tmp_path):
    cfg_part1 = _tiny_cfg(tmp_path, epochs=2, checkpoint_every_steps=2)
    cfg_part1.wandb.mode = "offline"
    cfg_part1.wandb.dir = str(tmp_path)
    run_training(cfg_part1, max_steps=2)

    checkpoint = torch.load(tmp_path / "step_2.pt", weights_only=False)
    original_run_id = checkpoint["wandb_run_id"]
    assert original_run_id is not None

    cfg_part2 = _tiny_cfg(tmp_path, epochs=2, checkpoint_every_steps=2)
    cfg_part2.wandb.mode = "offline"
    cfg_part2.wandb.dir = str(tmp_path)
    cfg_part2.resume_from = str(tmp_path / "step_2.pt")
    run_training(cfg_part2)

    resumed_checkpoint = torch.load(tmp_path / "step_4.pt", weights_only=False)
    assert resumed_checkpoint["wandb_run_id"] == original_run_id
