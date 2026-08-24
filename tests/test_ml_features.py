"""Tests for benchmarks/ml_features.py's feature engineering, on small synthetic chips."""

import numpy as np
import pytest

from benchmarks.ml_features import (
    FEATURE_NAMES_EVENT,
    FEATURE_NAMES_FULL,
    FEATURE_NAMES_RICH,
    build_event_relative_features,
    build_full_features,
    build_rich_features,
    fit_fill_values,
    impute_with_fill,
)


def _synthetic_chip(size=20):
    rng = np.random.default_rng(0)
    image = rng.normal(-15.0, 3.0, size=(3, size, size)).astype(np.float32)
    slope = rng.uniform(0, 10, size=(size, size)).astype(np.float32)
    hand = rng.uniform(0, 20, size=(size, size)).astype(np.float32)
    return image, slope, hand


def test_build_rich_features_shape_and_names_agree():
    image, slope, hand = _synthetic_chip()
    features = build_rich_features(image, slope, hand)

    assert features.shape == (400, len(FEATURE_NAMES_RICH))
    assert features.dtype == np.float32


def test_build_rich_features_rejects_shape_mismatch():
    image, slope, hand = _synthetic_chip()
    with pytest.raises(ValueError, match="shape mismatch"):
        build_rich_features(image, slope[:-1], hand)


def test_build_rich_features_base_channels_pass_through_unmodified():
    """The first 5 columns must be the raw VV/VH/ratio/slope/HAND -- no filtering."""
    image, slope, hand = _synthetic_chip()
    features = build_rich_features(image, slope, hand)

    assert np.allclose(features[:, 0], image[0].reshape(-1))
    assert np.allclose(features[:, 1], image[1].reshape(-1))
    assert np.allclose(features[:, 2], image[2].reshape(-1))
    assert np.allclose(features[:, 3], slope.reshape(-1))
    assert np.allclose(features[:, 4], hand.reshape(-1))


def test_local_std_is_near_zero_on_a_flat_chip():
    """A perfectly flat band should have ~zero local standard deviation everywhere."""
    image = np.full((3, 20, 20), -15.0, dtype=np.float32)
    slope = np.zeros((20, 20), dtype=np.float32)
    hand = np.zeros((20, 20), dtype=np.float32)

    features = build_rich_features(image, slope, hand)
    vh_std15_idx = FEATURE_NAMES_RICH.index("VH_std15")

    assert np.allclose(features[:, vh_std15_idx], 0.0, atol=1e-4)


def test_rich_features_handle_nan_border_without_propagating_everywhere():
    """A NaN border (like data/hand.py's flow-routing edge effect) must stay local."""
    image, slope, hand = _synthetic_chip()
    hand = hand.copy()
    hand[0, :] = np.nan  # a NaN row, like a real chip's border

    features = build_rich_features(image, slope, hand)

    # The HAND column itself carries NaN only where the input did.
    hand_idx = FEATURE_NAMES_RICH.index("HAND")
    reshaped = features[:, hand_idx].reshape(20, 20)
    assert np.isnan(reshaped[0, :]).all()
    assert not np.isnan(reshaped[1:, :]).any()

    # But a texture feature computed from VH (unaffected input) must NOT have
    # picked up NaN just because HAND was NaN elsewhere in the same chip.
    vh_std15_idx = FEATURE_NAMES_RICH.index("VH_std15")
    assert np.isfinite(features[:, vh_std15_idx]).all()


def test_build_event_relative_features_shape_and_sign():
    image, _, _ = _synthetic_chip()
    stats = {
        "vv_median": -15.0,
        "vh_median": -15.0,
        "vh_mean": -15.0,
        "vh_std": 3.0,
        "vh_otsu": -15.0,
    }

    features = build_event_relative_features(image, stats)

    assert features.shape == (400, len(FEATURE_NAMES_EVENT))
    # A pixel brighter than the event median should have a positive
    # VH_minus_event_median, a dimmer one negative -- direction sanity check.
    vh_minus_median_idx = FEATURE_NAMES_EVENT.index("VH_minus_event_median")
    expected = image[1].reshape(-1) - stats["vh_median"]
    assert np.allclose(features[:, vh_minus_median_idx], expected)


def test_build_full_features_concatenates_rich_and_event():
    image, slope, hand = _synthetic_chip()
    stats = {
        "vv_median": -15.0,
        "vh_median": -15.0,
        "vh_mean": -15.0,
        "vh_std": 3.0,
        "vh_otsu": -15.0,
    }

    full = build_full_features(image, slope, hand, stats)
    rich = build_rich_features(image, slope, hand)
    event = build_event_relative_features(image, stats)

    assert full.shape == (400, len(FEATURE_NAMES_FULL))
    assert np.allclose(full[:, : len(FEATURE_NAMES_RICH)], rich)
    assert np.allclose(full[:, len(FEATURE_NAMES_RICH) :], event)


def test_impute_with_fill_replaces_only_non_finite_values():
    features = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    fill = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    imputed = impute_with_fill(features, fill)

    assert np.allclose(imputed, [[1.0, 20.0, 3.0], [10.0, 5.0, 6.0]])
    # The original array must be untouched (impute_with_fill copies).
    assert np.isnan(features[0, 1])


def test_fit_fill_values_is_the_per_column_median_of_finite_values():
    features = np.array(
        [[1.0, np.nan], [2.0, 10.0], [3.0, 20.0], [np.nan, 30.0]], dtype=np.float32
    )

    fill = fit_fill_values(features)

    assert fill[0] == pytest.approx(2.0)
    assert fill[1] == pytest.approx(20.0)


def test_fit_fill_values_never_returns_nan_even_if_a_column_is_all_nan():
    features = np.array([[np.nan], [np.nan]], dtype=np.float32)
    fill = fit_fill_values(features)
    assert np.isfinite(fill).all()
