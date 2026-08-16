"""
Loss functions (spec section 4.2): Dice + BCE combo as the primary training
loss, with Focal loss kept available for the ablation the same section
calls for. Built on segmentation_models_pytorch's (smp) loss
implementations rather than reimplemented from scratch, for the same
reason models/unet.py doesn't reimplement U-Net -- ignore-index-aware
Dice/BCE/Focal for binary segmentation has already been solved and tested
upstream.

All losses here respect the data contract's ignore label
(data.contract.LABEL_IGNORE, = 255): no-data pixels (real Sen1Floods11
chips have scene-edge NaNs -- see VarunaNet_Spec.md's "known real-data
quirks") are excluded from the loss entirely, not counted as "non-water".
Getting this wrong is exactly the kind of silent bug the data contract
module warns about: a model "wrong" only on no-data pixels would still
pass a visual sanity check.

One thing worth recording here because it wasn't obvious from smp's docs
and had to be verified directly against its source: smp's DiceLoss and
FocalLoss both handle ignore_index correctly for binary segmentation
(DiceLoss zeroes ignored pixels out of both the prediction and target
before summing, so they drop out of the intersection/union ratio cleanly;
FocalLoss filters ignored pixels out entirely before its own mean).
smp's SoftBCEWithLogitsLoss does not: it zeroes the per-pixel loss at
ignored positions but then, with reduction="mean", still divides by the
*total* pixel count rather than the count of valid pixels. That silently
shrinks the loss for any chip with more no-data pixels, in proportion to
how much of it is no-data -- confirmed empirically before writing this:
a chip with 20/32 valid pixels came back with smp's raw mean at 0.497 vs.
a true masked mean of 0.796 for the same tensors. MaskedBCEWithLogitsLoss
below fixes that by summing (reusing smp's own masking) and dividing by
the valid pixel count itself, rather than reimplementing the masking too.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from data.contract import LABEL_IGNORE


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits, averaged only over non-ignored
    pixels. See the module docstring for why this exists instead of using
    smp.losses.SoftBCEWithLogitsLoss(reduction="mean") directly.
    """

    def __init__(self, ignore_index: int = LABEL_IGNORE):
        super().__init__()
        self.ignore_index = ignore_index
        # reduction="sum" so this class controls the normalization step;
        # smp still does the masking (zeroing ignored pixels' contribution).
        self._bce_sum = smp.losses.SoftBCEWithLogitsLoss(ignore_index=ignore_index, reduction="sum")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, 1, H, W] raw model output (see models/unet.py)
        targets: [B, H, W] int64, values in {0, 1, LABEL_IGNORE}
        """
        # F.binary_cross_entropy_with_logits requires logits and targets
        # to be the exact same shape (it does not broadcast a size-1
        # channel dim away), so squeeze the model's channel dim rather
        # than unsqueeze targets -- confirmed by testing this directly:
        # passing [B,1,H,W] against [B,H,W] raises a shape error rather
        # than silently broadcasting to something wrong.
        total = self._bce_sum(logits.squeeze(1), targets.float())
        valid_pixels = (targets != self.ignore_index).sum().clamp_min(1)
        return total / valid_pixels


class DiceBCELoss(nn.Module):
    """
    Dice + BCE combo loss, the spec's recommended default for imbalanced
    binary segmentation (flood water is a small minority of pixels in most
    scenes -- spec section 1). BCE gives a stable, well-behaved per-pixel
    gradient signal early in training when the model's predictions are
    still far from correct; Dice pushes directly toward the region-overlap
    metric (IoU-like) actually reported in benchmarks/RESULTS.md, which
    BCE alone doesn't optimize for.

    dice_weight/bce_weight are exposed so the training config can tune the
    balance between the two terms without editing this file -- useful for
    the loss ablation weeks 6-8 calls for.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        ignore_index: int = LABEL_IGNORE,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True, ignore_index=ignore_index)
        self.bce = MaskedBCEWithLogitsLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, 1, H, W] raw model output
        targets: [B, H, W] int64, values in {0, 1, LABEL_IGNORE}. Passed
                 to DiceLoss as-is (no squeeze needed -- it reshapes via
                 .view() internally rather than relying on broadcasting,
                 so a [B,1,H,W]-vs-[B,H,W] element-count match is enough).
        """
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(
            logits, targets
        )


def build_focal_loss(
    gamma: float = 2.0,
    alpha: float | None = 0.25,
    ignore_index: int = LABEL_IGNORE,
) -> nn.Module:
    """
    Build the Focal loss used for the loss ablation (spec section 4.2):
    "ablate against Focal loss". Used directly from smp rather than
    wrapped in a custom class, because -- unlike SoftBCEWithLogitsLoss --
    its binary-mode ignore_index handling filters ignored pixels out
    before its own mean reduction, so there's no dilution bug to fix here
    (verified by reading smp.losses.FocalLoss.forward directly rather than
    assumed). It also accepts [B,1,H,W] logits against [B,H,W] targets
    without a shape error, since it flattens both with .view(-1) rather
    than relying on broadcasting.

    gamma controls how much the loss down-weights already-easy (correctly
    classified) pixels; alpha reweights the positive (water) class, which
    matters here given severe class imbalance (spec section 1).
    """
    return smp.losses.FocalLoss(mode="binary", gamma=gamma, alpha=alpha, ignore_index=ignore_index)


def build_loss(name: str, **kwargs) -> nn.Module:
    """
    Config-driven loss selection, so the training config (Phase 3's
    Hydra/YAML config, per spec section 8) can pick a loss by name rather
    than the training loop hardcoding which one runs.
    """
    if name == "dice_bce":
        return DiceBCELoss(**kwargs)
    if name == "focal":
        return build_focal_loss(**kwargs)
    raise ValueError(f"unknown loss name {name!r}, expected 'dice_bce' or 'focal'")
