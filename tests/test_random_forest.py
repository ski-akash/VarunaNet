"""
Tests for the Random Forest baseline's feature building, class-balanced
sampling, training, and prediction, using small synthetic chips.
"""

import numpy as np
import pytest

from benchmarks.random_forest import (
    build_pixel_features,
    predict_water_mask,
    sample_training_pixels,
    train_random_forest,
)
from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER


def test_build_pixel_features_shape_and_order():
    image = np.zeros((3, 4, 4), dtype=np.float32)
    image[0], image[1], image[2] = 1.0, 2.0, 3.0
    slope = np.full((4, 4), 4.0, dtype=np.float32)
    hand = np.full((4, 4), 5.0, dtype=np.float32)

    features = build_pixel_features(image, slope, hand)

    assert features.shape == (16, 5)
    assert np.allclose(features[0], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_build_pixel_features_rejects_shape_mismatch():
    image = np.zeros((3, 4, 4), dtype=np.float32)
    slope = np.zeros((4, 5), dtype=np.float32)
    hand = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        build_pixel_features(image, slope, hand)


def test_sample_training_pixels_balances_classes():
    # 90 non-water pixels, 10 water pixels -- a deliberately imbalanced chip.
    label = np.full((10, 10), LABEL_NON_WATER, dtype=np.int64)
    label.reshape(-1)[:10] = LABEL_WATER
    features = np.zeros((100, 5), dtype=np.float32)
    rng = np.random.default_rng(seed=0)

    sampled_features, sampled_labels = sample_training_pixels(
        features, label, rng, samples_per_class=10
    )

    assert (sampled_labels == LABEL_WATER).sum() == 10
    assert (sampled_labels == LABEL_NON_WATER).sum() == 10


def test_sample_training_pixels_excludes_ignore_and_nan():
    label = np.full((4, 4), LABEL_NON_WATER, dtype=np.int64)
    label.reshape(-1)[:4] = LABEL_WATER
    label[0, 0] = LABEL_IGNORE  # would otherwise be a water pixel

    features = np.zeros((16, 5), dtype=np.float32)
    features[1] = np.nan  # would otherwise be a valid water pixel

    rng = np.random.default_rng(seed=0)
    sampled_features, sampled_labels = sample_training_pixels(
        features, label, rng, samples_per_class=10
    )

    # Only 2 of the original 4 water pixels are actually usable.
    assert (sampled_labels == LABEL_WATER).sum() == 2
    assert np.isfinite(sampled_features).all()


def test_predict_water_mask_treats_nan_features_as_non_water():
    features = np.zeros((4, 5), dtype=np.float32)
    features[2] = np.nan

    class StubModel:
        def predict(self, x):
            return np.full(len(x), LABEL_WATER)

    mask = predict_water_mask(StubModel(), features, shape=(2, 2))

    assert mask.shape == (2, 2)
    assert mask.dtype == bool
    assert mask.reshape(-1)[2] == False  # noqa: E712 -- clearer as a literal here
    assert mask.reshape(-1)[[0, 1, 3]].all()


def test_train_and_predict_separates_synthetic_classes():
    rng = np.random.default_rng(seed=0)
    n = 200
    water_features = rng.normal(loc=[-20, -25, 0.8, 1, 1], scale=0.5, size=(n, 5))
    land_features = rng.normal(loc=[-8, -12, 0.6, 10, 20], scale=0.5, size=(n, 5))

    training_features = np.concatenate([water_features, land_features]).astype(np.float32)
    training_labels = np.concatenate([np.full(n, LABEL_WATER), np.full(n, LABEL_NON_WATER)])

    model = train_random_forest(training_features, training_labels, n_estimators=20, seed=0)

    test_water = np.array([[-20, -25, 0.8, 1, 1]], dtype=np.float32)
    test_land = np.array([[-8, -12, 0.6, 10, 20]], dtype=np.float32)

    assert model.predict(test_water)[0] == LABEL_WATER
    assert model.predict(test_land)[0] == LABEL_NON_WATER
