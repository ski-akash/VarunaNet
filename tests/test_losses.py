"""
Tests for models/losses.py.

The behaviors under test here were verified empirically against smp's
actual source before models/losses.py was written (see that file's module
docstring) -- these tests encode those same findings as regression checks,
so a future smp version change or an accidental "simplification" of
MaskedBCEWithLogitsLoss back to smp's plain mean-reduction would be caught
immediately instead of silently reintroducing the dilution bug.

All tensors are tiny and synthetic per spec section 8 -- this file is
about verifying the loss math, not about real chip data.
"""

import torch
import torch.nn.functional as F

from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER
from models.losses import (
    DiceBCELoss,
    MaskedBCEWithLogitsLoss,
    build_focal_loss,
    build_loss,
    build_lovasz_loss,
    build_tversky_loss,
)


def _random_logits_and_targets(batch=2, h=4, w=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, 1, h, w, generator=g)
    targets = torch.randint(LABEL_NON_WATER, LABEL_WATER + 1, (batch, h, w), generator=g).long()
    return logits, targets


def test_dice_bce_forward_returns_scalar():
    logits, targets = _random_logits_and_targets()
    loss_fn = DiceBCELoss()
    loss = loss_fn(logits, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_masked_bce_matches_true_valid_pixel_average():
    # Same construction used to verify the dilution bug directly against
    # smp before writing the wrapper: 2 samples of 4x4, sample 1 has 3 of
    # 4 columns marked ignore, so only 8/32 total pixels are valid.
    logits = torch.randn(2, 4, 4)
    targets = torch.zeros(2, 4, 4)
    targets[0] = 1
    targets[1] = 1
    targets[1, :, :3] = LABEL_IGNORE

    loss_fn = MaskedBCEWithLogitsLoss(ignore_index=LABEL_IGNORE)
    actual = loss_fn(logits.unsqueeze(1), targets.long())

    valid_mask = targets != LABEL_IGNORE
    per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected = (per_pixel * valid_mask).sum() / valid_mask.sum()

    assert torch.isclose(actual, expected, atol=1e-6)


def test_masked_bce_diverges_from_naive_full_count_mean():
    # Documents *why* MaskedBCEWithLogitsLoss exists: smp's own
    # reduction="mean" divides by the total pixel count even after
    # masking, which measurably under-counts the loss whenever a chip has
    # a meaningful fraction of no-data pixels (real Sen1Floods11 chips do,
    # per VarunaNet_Spec.md's known real-data quirks).
    import segmentation_models_pytorch as smp

    logits = torch.randn(2, 4, 4)
    targets = torch.zeros(2, 4, 4)
    targets[0] = 1
    targets[1] = 1
    targets[1, :, :3] = LABEL_IGNORE

    correct = MaskedBCEWithLogitsLoss(ignore_index=LABEL_IGNORE)(
        logits.unsqueeze(1), targets.long()
    )
    naive = smp.losses.SoftBCEWithLogitsLoss(ignore_index=LABEL_IGNORE, reduction="mean")(
        logits, targets
    )

    assert not torch.isclose(correct, naive, atol=1e-3)


def test_dice_bce_ignores_predictions_at_ignored_pixels():
    # If ignore_index handling is working, the loss must not depend at
    # all on what the model predicted at ignored positions -- flipping
    # those logits to an extreme, obviously-wrong value should not move
    # the loss.
    logits, targets = _random_logits_and_targets()
    targets[:, 0, :] = LABEL_IGNORE  # first row of every sample is no-data

    loss_fn = DiceBCELoss()
    baseline = loss_fn(logits, targets)

    perturbed = logits.clone()
    perturbed[:, :, 0, :] = 100.0  # extreme logits, only at the ignored row
    perturbed_loss = loss_fn(perturbed, targets)

    assert torch.isclose(baseline, perturbed_loss, atol=1e-6)


def test_dice_bce_gradient_is_zero_at_ignored_pixels():
    logits, targets = _random_logits_and_targets()
    targets[:, 0, :] = LABEL_IGNORE
    logits.requires_grad_(True)

    loss_fn = DiceBCELoss()
    loss_fn(logits, targets).backward()

    ignored_grad = logits.grad[:, :, 0, :]
    valid_grad = logits.grad[:, :, 1:, :]

    assert torch.all(ignored_grad == 0)
    assert torch.any(valid_grad != 0)


def test_focal_loss_ignores_predictions_at_ignored_pixels():
    logits, targets = _random_logits_and_targets()
    targets[:, 0, :] = LABEL_IGNORE

    loss_fn = build_focal_loss()
    baseline = loss_fn(logits, targets)

    perturbed = logits.clone()
    perturbed[:, :, 0, :] = 100.0
    perturbed_loss = loss_fn(perturbed, targets)

    assert torch.isclose(baseline, perturbed_loss, atol=1e-6)


def test_focal_loss_accepts_channel_dim_logits_against_flat_targets():
    # Regression check for the shape handling documented in
    # build_focal_loss's docstring: [B,1,H,W] logits against [B,H,W]
    # targets should just work (smp flattens both with .view(-1)), unlike
    # SoftBCEWithLogitsLoss which requires an exact shape match.
    logits, targets = _random_logits_and_targets()
    loss_fn = build_focal_loss()
    loss = loss_fn(logits, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_tversky_loss_ignores_predictions_at_ignored_pixels():
    # Inherited from smp.losses.DiceLoss's masking (see build_tversky_loss's
    # docstring) -- same check as Dice/Focal above.
    logits, targets = _random_logits_and_targets()
    targets[:, 0, :] = LABEL_IGNORE

    loss_fn = build_tversky_loss()
    baseline = loss_fn(logits, targets)

    perturbed = logits.clone()
    perturbed[:, :, 0, :] = 100.0
    perturbed_loss = loss_fn(perturbed, targets)

    assert torch.isclose(baseline, perturbed_loss, atol=1e-6)


def test_tversky_loss_reduces_to_dice_at_alpha_equals_beta():
    # Documented directly in smp's own docstring (alpha == beta == 0.5 ==
    # DiceLoss) -- verified here rather than trusted, since it's the
    # cheapest possible check that build_tversky_loss is wired to the
    # right smp class with the right defaults threaded through.
    logits, targets = _random_logits_and_targets()

    tversky_at_half = build_tversky_loss(alpha=0.5, beta=0.5, gamma=1.0)(logits, targets)
    dice = DiceBCELoss(dice_weight=1.0, bce_weight=0.0)(logits, targets)
    # DiceBCELoss's dice term is smp.losses.DiceLoss itself, not wrapped
    # further, so this should match tightly rather than approximately.
    dice_only = torch.nn.functional.relu(dice)  # dice term is already >= 0
    assert torch.isclose(tversky_at_half, dice_only, atol=1e-5)


def test_tversky_loss_favors_recall_when_beta_exceeds_alpha():
    # A prediction that under-calls water (false negatives) should be
    # penalized more than one that over-calls water (false positives) by
    # the same amount, once beta > alpha -- the whole reason this
    # project's default (0.3/0.7) was chosen over plain Dice.
    targets = torch.zeros(1, 4, 4)
    targets[0, :2, :] = LABEL_WATER  # top half is water

    # false-negative logits: confidently predicts non-water everywhere,
    # missing all real water.
    fn_logits = torch.full((1, 1, 4, 4), -10.0)
    # false-positive logits: confidently predicts water everywhere,
    # over-calling the bottom (non-water) half by the same margin.
    fp_logits = torch.full((1, 1, 4, 4), 10.0)

    loss_fn = build_tversky_loss(alpha=0.3, beta=0.7)
    fn_loss = loss_fn(fn_logits, targets.long())
    fp_loss = loss_fn(fp_logits, targets.long())

    assert fn_loss > fp_loss


def test_lovasz_loss_ignores_predictions_at_ignored_pixels():
    logits, targets = _random_logits_and_targets()
    targets[:, 0, :] = LABEL_IGNORE

    loss_fn = build_lovasz_loss()
    baseline = loss_fn(logits, targets)

    perturbed = logits.clone()
    perturbed[:, :, 0, :] = 100.0
    perturbed_loss = loss_fn(perturbed, targets)

    assert torch.isclose(baseline, perturbed_loss, atol=1e-6)


def test_lovasz_loss_accepts_channel_dim_logits_against_flat_targets():
    # Regression check for the shape claim in build_lovasz_loss's
    # docstring: [B,1,H,W] logits against [B,H,W] targets works because
    # both are flattened with .view(-1) before use, so squeezing the
    # size-1 channel dim first isn't necessary -- confirmed with a real
    # forward/backward pass, not just read from source.
    logits, targets = _random_logits_and_targets()
    logits.requires_grad_(True)
    loss_fn = build_lovasz_loss()
    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_build_loss_dispatches_by_name():
    dice_bce = build_loss("dice_bce")
    assert isinstance(dice_bce, DiceBCELoss)

    logits, targets = _random_logits_and_targets()
    focal = build_loss("focal")
    assert torch.isfinite(focal(logits, targets))

    tversky = build_loss("tversky")
    assert torch.isfinite(tversky(logits, targets))

    lovasz = build_loss("lovasz")
    assert torch.isfinite(lovasz(logits, targets))


def test_build_loss_rejects_unknown_name():
    try:
        build_loss("not_a_real_loss")
    except ValueError:
        return
    raise AssertionError("expected build_loss to raise ValueError for an unknown name")
