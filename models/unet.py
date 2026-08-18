"""
The primary model (spec section 4.1): U-Net with a ResNet-34 encoder,
initialized from ImageNet-pretrained weights.

We don't hand-roll U-Net. `segmentation_models_pytorch` (smp) already
implements the encoder/decoder plumbing correctly and gives us the same
architecture family (U-Net++, DeepLabV3+, ...) needed later for Phase 4's
architecture study, all through one consistent interface. That means the
effort here goes into wiring it to *our* data contract, not re-deriving a
segmentation architecture that's been solved many times over.

Two things make this model non-standard relative to a typical smp tutorial:

1. Input channels. Our data contract (data/contract.py) is 5 channels
   (VV_db, VH_db, VV_VH_ratio, slope, HAND), not the 3-channel RGB that
   ImageNet pretraining expects. smp handles this itself: when
   `in_channels != 3`, it adapts the pretrained first convolution by
   averaging the original 3-channel kernel weights across the new channel
   count, rather than discarding the pretrained weights entirely. So we
   still benefit from ImageNet-pretrained low-level features (edges,
   textures) even though 2 of our 5 channels are radar backscatter and 2
   are terrain derivatives, not color.

2. Output channels. This is binary segmentation (water vs non-water), so
   the model outputs a single logit channel per pixel rather than a
   per-class channel. A sigmoid (applied by the loss function, not here --
   see models/losses.py) turns that logit into a water probability. Kept
   as raw logits rather than probabilities because BCEWithLogitsLoss
   (used inside the Dice+BCE combo loss) is numerically more stable than
   applying sigmoid ourselves and then computing plain BCE.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

from data.contract import NUM_CHANNELS

# Binary segmentation: one logit per pixel (water vs non-water), not one
# channel per class. See module docstring point 2.
NUM_OUTPUT_CLASSES = 1


_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
}


def build_unet(
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = NUM_CHANNELS,
    classes: int = NUM_OUTPUT_CLASSES,
) -> nn.Module:
    """
    Build the primary model: U-Net (or, per Phase 4's architecture study,
    U-Net++) with the given encoder (default ResNet-34, per spec section
    4.1).

    Arguments are exposed rather than hardcoded so this same function
    covers the architecture study -- e.g. swapping encoder_name to try a
    deeper/shallower backbone, setting encoder_weights=None to compare
    against training from scratch, or architecture="unetplusplus" to
    compare against U-Net's nested skip connections -- without touching
    this file again.

    Returns a plain nn.Module that maps an input tensor of shape
    [B, in_channels, H, W] to an output tensor of shape [B, classes, H, W]
    of raw (pre-sigmoid) logits. H and W must each be divisible by 32:
    a ResNet encoder downsamples 5 times (2^5 = 32), and the decoder
    upsamples back by the same factor at each stage, so any input size
    not evenly divisible by that would leave the encoder and decoder
    output resolutions mismatched at the final skip connection. Both
    architectures share this constraint -- U-Net++'s extra nested skip
    connections don't change the encoder/decoder downsampling factor.
    """
    if architecture not in _ARCHITECTURES:
        raise ValueError(
            f"architecture {architecture!r} isn't wired up yet -- expected one of "
            f"{sorted(_ARCHITECTURES)}"
        )
    model_cls = _ARCHITECTURES[architecture]
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
