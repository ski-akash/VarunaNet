"""
Per-channel normalization statistics (mean and standard deviation).

These must be computed once, only on the training split, and then reused
unchanged everywhere else (validation, test, and later, live inference on
real Sentinel-1 scenes). If serving code ever recomputes its own stats
instead of loading the ones saved during training, the model sees inputs
scaled differently than what it was trained on and predictions quietly
degrade. Persisting these to a file, and loading that same file everywhere,
is what prevents that.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data.contract import CHANNELS, NUM_CHANNELS


@dataclass
class NormalizationStats:
    # channels records which channel each mean/std entry belongs to, so a
    # stats file computed for one channel order can never be silently
    # applied to data in a different channel order.
    channels: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        raw = json.loads(Path(path).read_text())
        return cls(
            channels=tuple(raw["channels"]),
            mean=tuple(raw["mean"]),
            std=tuple(raw["std"]),
        )


def compute_normalization_stats(tensors: list[np.ndarray]) -> NormalizationStats:
    """
    Compute per-channel mean/std over a list of [C, H, W] input tensors.

    Only ever call this on the training split - computing stats on
    validation or test data would leak information about those splits into
    the model.

    NaN-aware (np.nanmean/np.nanstd) rather than plain mean/std: real
    chips carry NaN pixels by design, not just as a rare edge case --
    data/hand.py's HAND computation leaves the outer border of every chip
    as NaN (a documented, unavoidable flow-routing edge effect), so every
    single training chip contributes at least some NaN pixels to the HAND
    channel. Plain np.mean/np.std propagate a single NaN into a NaN result
    for the *entire* channel (confirmed directly: np.array([1., 2.,
    np.nan]).mean() is nan, not "mean of the finite values"), which would
    have silently produced NaN normalization stats -- and therefore NaN
    everywhere after every future apply_normalization() call -- the very
    first time this was run against the real dataset.
    """
    if not tensors:
        raise ValueError("cannot compute normalization stats from an empty list")

    # Stack into [N, C, H, W] then flatten each channel across every image
    # and every pixel, so mean/std are computed per channel.
    stacked = np.stack(tensors, axis=0)
    if stacked.shape[1] != NUM_CHANNELS:
        raise ValueError(f"expected {NUM_CHANNELS} channels, got {stacked.shape[1]}")

    per_channel = stacked.transpose(1, 0, 2, 3).reshape(NUM_CHANNELS, -1)
    mean = np.nanmean(per_channel, axis=1)
    std = np.nanstd(per_channel, axis=1)

    return NormalizationStats(
        channels=CHANNELS,
        mean=tuple(mean.tolist()),
        std=tuple(std.tolist()),
    )


def apply_normalization(tensor: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply saved per-channel (x - mean) / std normalization to a [C, H, W] tensor."""
    if stats.channels != CHANNELS:
        raise ValueError(
            f"stats were computed for channels {stats.channels}, "
            f"but the current data contract is {CHANNELS}"
        )

    mean = np.array(stats.mean, dtype=np.float32).reshape(NUM_CHANNELS, 1, 1)
    std = np.array(stats.std, dtype=np.float32).reshape(NUM_CHANNELS, 1, 1)
    return ((tensor - mean) / std).astype(np.float32)
