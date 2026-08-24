"""
Tests for training/checkpoint.py's RNG state restoration.

Regression coverage for a real bug: torch.load(..., map_location="cuda")
applies that map_location to EVERY tensor in a checkpoint, including the RNG
state bytes -- not just the model weights, which is the only thing anyone
expects it to move. torch.set_rng_state then rejects the CUDA-resident
tensor with "RNG state must be a torch.ByteTensor", since RNG state has to
be a CPU byte buffer regardless of what device training runs on. Confirmed
live on a real A100 node running training/evaluate_test.py against a real
checkpoint -- not a hypothetical -- and via a direct repro showing the exact
same tensor works once moved back to CPU with .cpu().

This means SLURM's --requeue auto-resume (training/train.py's real resume
path, which calls load_checkpoint with map_location=device) would hit this
exact crash on every GPU run, silently defeating the resume-exactness
guarantee training/test_train.py's own resume tests exist to protect -- it
went uncaught there only because every local test runs on CPU, where
map_location="cpu" is already a no-op for tensor placement.

HONEST LIMITATION: this machine and the cluster's login node both lack a
GPU, so the actual CUDA-tensor-rejected-by-isinstance-check failure cannot
be reproduced in an automated test here, the same class of gap this
project already documents elsewhere (e.g. infra/Dockerfile.train not being
locally buildable). What CAN be tested without a GPU is that
_restore_rng_state calls .cpu() on every RNG tensor before handing it to
torch's restore functions -- exercised below with mocks standing in for
torch.set_rng_state / torch.cuda.set_rng_state_all, so a future
simplification that drops those .cpu() calls fails this test immediately
rather than only failing months later on a real GPU node.
"""

from unittest.mock import MagicMock, patch

from training.checkpoint import _restore_rng_state


def test_restore_rng_state_moves_torch_state_to_cpu_before_restoring():
    torch_state = MagicMock(name="torch_rng_state")
    rng_state = {
        "torch": torch_state,
        "cuda": None,
        "numpy": ("MT19937", [0], 0, 0, 0.0),
        "python": (3, (0,) * 625, None),
    }

    with (
        patch("training.checkpoint.torch.set_rng_state") as set_rng_state,
        patch("training.checkpoint.np.random.set_state"),
        patch("training.checkpoint.random.setstate"),
    ):
        _restore_rng_state(rng_state)

        # .cpu() must be called on the raw captured tensor, and its RESULT --
        # not the original, possibly-CUDA-resident tensor -- is what actually
        # reaches torch.set_rng_state.
        torch_state.cpu.assert_called_once()
        set_rng_state.assert_called_once_with(torch_state.cpu.return_value)


def test_restore_rng_state_moves_every_cuda_device_state_to_cpu():
    device0_state = MagicMock(name="cuda_device_0_rng_state")
    device1_state = MagicMock(name="cuda_device_1_rng_state")
    rng_state = {
        "torch": MagicMock(),
        "cuda": [device0_state, device1_state],
        "numpy": ("MT19937", [0], 0, 0, 0.0),
        "python": (3, (0,) * 625, None),
    }

    with (
        patch("training.checkpoint.torch.set_rng_state"),
        patch("training.checkpoint.torch.cuda.is_available", return_value=True),
        patch("training.checkpoint.torch.cuda.set_rng_state_all") as set_rng_state_all,
        patch("training.checkpoint.np.random.set_state"),
        patch("training.checkpoint.random.setstate"),
    ):
        _restore_rng_state(rng_state)

        device0_state.cpu.assert_called_once()
        device1_state.cpu.assert_called_once()
        set_rng_state_all.assert_called_once_with(
            [device0_state.cpu.return_value, device1_state.cpu.return_value]
        )


def test_restore_rng_state_skips_cuda_restore_when_state_is_none():
    """No CUDA state was captured (e.g. checkpoint saved on CPU) -- must not crash."""
    rng_state = {
        "torch": MagicMock(),
        "cuda": None,
        "numpy": ("MT19937", [0], 0, 0, 0.0),
        "python": (3, (0,) * 625, None),
    }

    with (
        patch("training.checkpoint.torch.set_rng_state"),
        patch("training.checkpoint.torch.cuda.set_rng_state_all") as set_rng_state_all,
        patch("training.checkpoint.np.random.set_state"),
        patch("training.checkpoint.random.setstate"),
    ):
        _restore_rng_state(rng_state)

        set_rng_state_all.assert_not_called()
