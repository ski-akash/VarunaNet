"""
SAR speckle-noise augmentation (spec section 4.2: "speckle-noise
augmentation specifically -- this is SAR-appropriate and a nice
domain-aware detail").

SAR speckle is multiplicative, not additive, unlike ordinary optical-image
noise: each resolution cell contains many sub-wavelength scatterers whose
reflections interfere coherently, so the *true* backscatter power gets
multiplied by a random speckle factor rather than having noise added on
top. For an L-look intensity image, the standard model is
observed_power = true_power * eta, with eta ~ Gamma(shape=L, scale=1/L)
(mean 1, variance 1/L -- fewer looks means a grainier image). Applied in
linear power, not dB: multiplying dB values directly has no physical
meaning, since dB is already log-compressed.

Uses numpy's legacy global RNG (np.random.gamma, not
np.random.default_rng()) deliberately -- training/checkpoint.py's
seed_everything/save_checkpoint/load_checkpoint capture and restore
np.random.get_state()/set_state() (the legacy global generator's state),
not an independent Generator instance's state, so this has to draw from
the same global stream to participate in this project's existing
determinism and resume-exactness guarantees.
"""

import numpy as np


def apply_speckle_noise(band_db: np.ndarray, looks: float) -> np.ndarray:
    """
    band_db: dB backscatter values (VV_db or VH_db), any shape. looks:
    simulated number of looks -- lower means noisier (speckle variance is
    1/looks). NaN pixels pass through as NaN (NaN * anything is NaN, so
    no special-casing is needed for the real scene-edge no-data pixels
    this project's chips carry).
    """
    linear = 10.0 ** (band_db / 10.0)
    speckle = np.random.gamma(shape=looks, scale=1.0 / looks, size=band_db.shape)
    noisy_linear = linear * speckle
    return 10.0 * np.log10(noisy_linear)
