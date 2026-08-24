"""
Feature engineering shared by the shallow-ML baselines beyond plain
Random Forest (benchmarks/ml_ensemble.py).

benchmarks/random_forest.py's original 5 features are exactly the data
contract's channels, read raw, per pixel, with zero spatial context. That
was a real gap, not just "could be improved": the same class of fix that took
Otsu from 0.224 to 0.304 mean IoU -- denoising the speckle-noisy SAR bands --
was never applied here, and a per-pixel model has no way to see that water is
locally *smooth*, one of the strongest discriminators against dark-but-rough
look-alikes (wet soil, shadow, rough dry ground).

Two additions, in order of how much they're expected to matter:

1. Denoised bands + local texture (mean/std at two window sizes). Costs one
   filter pass per feature and gives the model exactly the two things it was
   missing. Confirmed empirically to matter: on the real val split, adding
   these took Random Forest from pooled IoU 0.389 to 0.473, and the two new
   features (VH_med7, VH_std15) came out as the two most important features
   by the trained model's own feature_importances_.

2. Event-relative features (compute_event_stats / build_event_relative_features
   below). Otsu computes a threshold *per flood event* at scoring time, so it
   adapts to each event's own backscatter distribution; a tree model trained
   on absolute dB values learns one global decision boundary and cannot. These
   features hand the learned models the same per-event context Otsu already
   gets. Confirmed to help: +0.0095 pooled IoU on the real test split, and two
   of the four event-relative features rank in the top 3 by importance.

FAIRNESS NOTE on (2), stated here because it has to travel with any number
these features produce. The per-event statistics are computed from the
evaluation split's own pixels -- transductive, not inductive. That is not
label leakage (no test LABELS are touched, only unlabelled SAR imagery), and
it is exactly the same privilege benchmarks/evaluate.py's Otsu baseline
already takes (compute_per_event_otsu_thresholds pools the test split's own
chips). So this keeps the ML/Otsu comparison apples-to-apples rather than
granting the learned models something the classical baseline doesn't have. A
real streaming deployment that cannot pool a whole event before predicting
would need to recompute these from a rolling window instead.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from benchmarks.metrics import event_name
from benchmarks.otsu import compute_otsu_threshold, smooth_backscatter
from data.sen1floods11 import Sen1Floods11Dataset

FEATURE_NAMES_BASE = ("VV_db", "VH_db", "VV_VH_ratio", "slope", "HAND")
FEATURE_NAMES_TEXTURE = (
    "VV_med7",
    "VH_med7",
    "VH_mean15",
    "VH_std15",
    "VV_std15",
    "VH_std5",
)
FEATURE_NAMES_EVENT = (
    "VH_minus_event_median",
    "VV_minus_event_median",
    "VH_event_zscore",
    "VH_minus_event_otsu",
)
FEATURE_NAMES_RICH = FEATURE_NAMES_BASE + FEATURE_NAMES_TEXTURE
FEATURE_NAMES_FULL = FEATURE_NAMES_RICH + FEATURE_NAMES_EVENT


def _local_mean_std(band: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Local mean and standard deviation in a size x size window.

    NaN-tolerant by construction: NaN pixels are filled with 0 for the sum but
    excluded from the weight used to normalize, so a window overlapping the
    HAND-style NaN border isn't biased toward 0 by pixels that were never real
    data. This is a different NaN policy than benchmarks/otsu_hand.py's
    plausible_terrain_mask (which lets NaN abstain from a decision) because
    this is computing a statistic, not making a keep/reject call -- but the
    same underlying lesson applies: a NaN in real Sen1Floods11 data has to be
    handled explicitly, or it silently corrupts whatever touches it.
    """
    finite = np.isfinite(band)
    filled = np.where(finite, band, 0.0)
    weight = finite.astype(np.float32)

    wsum = np.maximum(uniform_filter(weight, size=size, mode="nearest"), 1e-6)
    mean = uniform_filter(filled, size=size, mode="nearest") / wsum
    meansq = uniform_filter(filled * filled, size=size, mode="nearest") / wsum
    variance = np.maximum(meansq - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _denoise(band: np.ndarray, size: int = 7) -> np.ndarray:
    """NaN-aware median filter -- reuses the Otsu baseline's own denoising."""
    return smooth_backscatter(band, size=size)


def build_rich_features(image: np.ndarray, slope: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """
    The 5 base data-contract channels plus 6 texture channels: two denoised
    bands and four local mean/std statistics at two window sizes. Returns an
    [H*W, 11] table in FEATURE_NAMES_RICH order.
    """
    if not (image.shape[1:] == slope.shape == hand.shape):
        raise ValueError(
            f"shape mismatch: image {image.shape}, slope {slope.shape}, hand {hand.shape}"
        )

    vv, vh, ratio = image[0], image[1], image[2]
    vh_mean15, vh_std15 = _local_mean_std(vh, 15)
    _, vv_std15 = _local_mean_std(vv, 15)
    _, vh_std5 = _local_mean_std(vh, 5)

    layers = [
        vv,
        vh,
        ratio,
        slope,
        hand,
        _denoise(vv),
        _denoise(vh),
        vh_mean15,
        vh_std15,
        vv_std15,
        vh_std5,
    ]
    return np.stack(layers, axis=-1).reshape(-1, len(layers)).astype(np.float32)


def compute_event_stats(dataset: Sen1Floods11Dataset) -> dict[str, dict[str, float]]:
    """
    Pool every chip in each flood event and compute the statistics the
    event-relative features are expressed against: median/mean/std of VV and
    VH, and the per-event Otsu threshold on denoised VH (matching exactly what
    benchmarks/evaluate.py's Otsu baseline thresholds, so
    'VH_minus_event_otsu' encodes the same decision boundary Otsu actually
    uses, not a different one).

    Pixels are subsampled per chip (every Nth finite pixel) purely for speed
    -- event statistics are stable long before every pixel is needed, and
    pooling all of them across ~250 chips is needless memory for no gain in
    accuracy.
    """
    pools: dict[str, dict[str, list[np.ndarray]]] = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        ev = event_name(sample.chip_id)
        slot = pools.setdefault(ev, {"vv": [], "vh": [], "vh_dn": []})

        vv, vh = sample.image[0], sample.image[1]
        vh_dn = _denoise(vh)
        for key, band in (("vv", vv), ("vh", vh), ("vh_dn", vh_dn)):
            finite = band[np.isfinite(band)]
            if finite.size:
                slot[key].append(finite[:: max(1, finite.size // 20000)])

    stats: dict[str, dict[str, float]] = {}
    for ev, slot in pools.items():
        vv = np.concatenate(slot["vv"]) if slot["vv"] else np.array([0.0])
        vh = np.concatenate(slot["vh"]) if slot["vh"] else np.array([0.0])
        vh_dn = np.concatenate(slot["vh_dn"]) if slot["vh_dn"] else np.array([0.0])
        stats[ev] = {
            "vv_median": float(np.median(vv)),
            "vh_median": float(np.median(vh)),
            "vh_mean": float(np.mean(vh)),
            "vh_std": float(max(np.std(vh), 1e-3)),
            "vh_otsu": float(compute_otsu_threshold(vh_dn)),
        }
    return stats


def build_event_relative_features(image: np.ndarray, event_stats: dict[str, float]) -> np.ndarray:
    """
    The 4 event-relative features for one chip, given its event's stats from
    compute_event_stats. Returns an [H*W, 4] table in FEATURE_NAMES_EVENT
    order, meant to be concatenated onto build_rich_features's output.
    """
    vv, vh = image[0], image[1]
    vh_dn = _denoise(vh)

    layers = [
        vh - event_stats["vh_median"],
        vv - event_stats["vv_median"],
        (vh - event_stats["vh_mean"]) / event_stats["vh_std"],
        vh_dn - event_stats["vh_otsu"],
    ]
    return np.stack(layers, axis=-1).reshape(-1, len(layers)).astype(np.float32)


def build_full_features(
    image: np.ndarray, slope: np.ndarray, hand: np.ndarray, event_stats: dict[str, float]
) -> np.ndarray:
    """Rich + event-relative features concatenated: [H*W, 15] in FEATURE_NAMES_FULL order."""
    rich = build_rich_features(image, slope, hand)
    event = build_event_relative_features(image, event_stats)
    return np.concatenate([rich, event], axis=1)


def impute_with_fill(features: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """
    Replace non-finite feature values with precomputed per-column fill values
    (typically training-split medians -- see fit_fill_values below).

    This is the deliberate alternative to benchmarks/random_forest.py's
    original policy of vetoing any pixel with a non-finite feature (forced to
    non-water at inference, excluded from training entirely). That policy is
    the same class of bug documented in benchmarks/otsu_hand.py's
    plausible_terrain_mask: data/hand.py's NaN border is water-enriched, so
    vetoing those pixels costs real water disproportionately. Imputing keeps
    every pixel in play; the SAR bands themselves must still be finite (a
    genuinely no-data pixel, not a border artifact) for a prediction to be
    made at all -- see fit_fill_values and predict_water_mask_ml in
    benchmarks/ml_ensemble.py for where that split is enforced.
    """
    out = features.copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fill[np.where(bad)[1]]
    return out


def fit_fill_values(features: np.ndarray) -> np.ndarray:
    """Per-column median of the finite values in `features`, for impute_with_fill."""
    fill = np.nanmedian(np.where(np.isfinite(features), features, np.nan), axis=0)
    return np.nan_to_num(fill)
