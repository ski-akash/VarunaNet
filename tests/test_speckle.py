"""
Tests for data/speckle.py -- spec section 4.2's speckle-noise
augmentation. Uses np.random.seed for determinism (the same legacy
global RNG this module deliberately draws from, see its docstring), not
an injected Generator.
"""

import numpy as np

from data.speckle import apply_speckle_noise


def test_noise_actually_changes_values():
    np.random.seed(0)
    band_db = np.full((32, 32), -12.0, dtype=np.float64)

    noisy = apply_speckle_noise(band_db, looks=4.0)

    assert not np.allclose(noisy, band_db)


def test_nan_pixels_stay_nan():
    band_db = np.full((16, 16), -12.0, dtype=np.float64)
    band_db[0, 0] = np.nan

    noisy = apply_speckle_noise(band_db, looks=4.0)

    assert np.isnan(noisy[0, 0])
    assert not np.isnan(noisy[1, 1])


def test_mean_is_approximately_unbiased_in_linear_power():
    # The Gamma(shape=looks, scale=1/looks) speckle factor has mean 1 by
    # construction, so averaging many independent noisy draws of a
    # constant input, back in linear power, should converge close to the
    # original linear power -- confirms the noise doesn't systematically
    # brighten or darken the signal, only adds variance around it.
    np.random.seed(1)
    band_db = np.full((256, 256), -12.0, dtype=np.float64)
    true_linear = 10.0 ** (band_db[0, 0] / 10.0)

    noisy = apply_speckle_noise(band_db, looks=4.0)
    noisy_linear_mean = np.mean(10.0 ** (noisy / 10.0))

    assert abs(noisy_linear_mean - true_linear) / true_linear < 0.05


def test_fewer_looks_means_more_variance():
    # Lower `looks` -- fewer simulated radar looks -- should produce a
    # grainier (higher-variance) result, per the Gamma(shape=looks,
    # scale=1/looks) model's variance = 1/looks.
    np.random.seed(2)
    band_db = np.full((256, 256), -12.0, dtype=np.float64)

    noisy_strong = apply_speckle_noise(band_db, looks=1.0)
    noisy_weak = apply_speckle_noise(band_db, looks=16.0)

    assert np.std(noisy_strong) > np.std(noisy_weak)
