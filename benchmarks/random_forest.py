"""
Random Forest baseline: a per-pixel, non-deep, but genuinely *learned*
classifier -- the middle tier between the physics-only baselines (Otsu,
Otsu+HAND) and the CNN. Unlike those, this is trained on labeled data
using the full 5-channel feature set, so it's the first model in the
benchmark table that can answer "does learning help at all", separately
from "does deep learning specifically help".

Features per pixel: VV_db, VH_db, VV_VH_ratio, slope, HAND -- the same
five channels the data contract defines for the CNN, so this baseline is
judged on the same information the later models get.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER

FEATURE_NAMES = ("VV_db", "VH_db", "VV_VH_ratio", "slope", "HAND")

# A Sen1Floods11 chip is 512x512 = ~262k pixels; 431 training-split chips
# is ~113M pixels total, far more than a Random Forest needs and more
# than is practical to fit on a laptop. This caps how many pixels of each
# class get drawn per chip during training -- not a physical constant,
# just a practical budget.
DEFAULT_SAMPLES_PER_CLASS_PER_CHIP = 1000
DEFAULT_N_ESTIMATORS = 100


def build_pixel_features(image: np.ndarray, slope: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """
    Stack a chip's SAR channels with its terrain channels into a
    [n_pixels, 5] feature table, in the same channel order as
    FEATURE_NAMES. `image` is the [3, H, W] array from
    Sen1Floods11Dataset (VV_db, VH_db, VV_VH_ratio); slope/hand are each
    [H, W] and must already be on the same pixel grid as `image`.
    """
    if not (image.shape[1:] == slope.shape == hand.shape):
        raise ValueError(
            f"shape mismatch: image {image.shape}, slope {slope.shape}, hand {hand.shape}"
        )

    features = np.stack([image[0], image[1], image[2], slope, hand], axis=-1)
    return features.reshape(-1, len(FEATURE_NAMES)).astype(np.float32)


def sample_training_pixels(
    features: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS_PER_CHIP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw up to `samples_per_class` pixels from each of the water and
    non-water classes in one chip, skipping no-data (ignore-index) pixels
    and any pixel with a non-finite feature value (e.g. the HAND border
    artifact documented in data/hand.py). Sampling per class instead of
    uniformly at random keeps one heavily-flooded or entirely-dry chip
    from silently skewing the training set toward a single class -- the
    spec calls out severe class imbalance as a real problem here (SS1).
    Returns fewer than `samples_per_class` pixels for a class if the chip
    doesn't have that many valid pixels of it.
    """
    label_flat = label.reshape(-1)
    valid = (label_flat != LABEL_IGNORE) & np.isfinite(features).all(axis=1)

    sampled_features = []
    sampled_labels = []
    for class_value in (LABEL_WATER, LABEL_NON_WATER):
        class_indices = np.flatnonzero(valid & (label_flat == class_value))
        if class_indices.size == 0:
            continue
        n = min(samples_per_class, class_indices.size)
        chosen = rng.choice(class_indices, size=n, replace=False)
        sampled_features.append(features[chosen])
        sampled_labels.append(label_flat[chosen])

    if not sampled_features:
        return (
            np.empty((0, features.shape[1]), dtype=features.dtype),
            np.empty((0,), dtype=label.dtype),
        )

    return np.concatenate(sampled_features), np.concatenate(sampled_labels)


def train_random_forest(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seed: int = 0,
) -> RandomForestClassifier:
    """Fit a Random Forest on pre-sampled pixel features (see sample_training_pixels)."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    model.fit(training_features, training_labels)
    return model


def predict_water_mask(
    model: RandomForestClassifier, features: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """
    Predict a full water/non-water mask covering every pixel in a chip.
    Pixels with a non-finite feature value are conservatively predicted
    as non-water rather than passed to the model, the same convention
    used by the Otsu baselines for their own no-data pixels.
    """
    valid = np.isfinite(features).all(axis=1)
    predictions = np.zeros(features.shape[0], dtype=bool)
    if valid.any():
        predictions[valid] = model.predict(features[valid]) == LABEL_WATER
    return predictions.reshape(shape)
