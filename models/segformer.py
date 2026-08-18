"""
SegFormer (Xie et al. 2021) -- the transformer comparison Phase 4's
architecture study calls for (spec section 4.3): "is a CNN the right
choice here? Answering this with data, rather than assuming, is a
genuine signal of maturity." B0 (lightweight) and B2 (larger) variants,
both starting from ADE20K-finetuned segmentation weights.

Wraps HuggingFace's SegformerForSemanticSegmentation rather than
hand-rolling a vision transformer, for the same reason models/unet.py
wraps segmentation_models_pytorch instead of hand-rolling U-Net -- this
architecture has already been implemented and pretrained correctly
upstream.

Three adaptations needed to fit our data contract, mirroring
models/unet.py's own three:

1. Input channels. The pretrained patch embedding's first conv expects
   3 channels (RGB); ours is 5 (VV_db, VH_db, VV_VH_ratio, slope, HAND).
   Same fix as smp's U-Net: average the pretrained 3-channel kernel
   weights across the new channel count, rather than discarding the
   pretrained weights and starting that layer from scratch.

2. Output channels. HuggingFace's checkpoint is fine-tuned for ADE20K's
   150 semantic classes; ours is binary water segmentation, one logit
   per pixel. The final 1x1 classifier conv is replaced with a
   freshly-initialized single-output version -- there's no meaningful
   way to reuse 150-class classifier weights for a 1-class problem, so
   unlike the input-channel adaptation, this layer starts fresh.

3. Output resolution. SegFormer's decode head outputs at 1/4 of the
   input resolution (a 512x512 input yields 128x128 logits) instead of
   upsampling all the way back like U-Net's decoder does. Bilinear
   interpolation back to the input's H/W happens in forward() here, so
   this model's output shape matches every other model's contract
   ([B, classes, H, W]) without every caller needing to know SegFormer's
   internal resolution quirk.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

_CHECKPOINTS = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Same averaging trick smp uses for U-Net's first conv -- see module docstring point 1."""
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        averaged = conv.weight.mean(dim=1, keepdim=True)  # [out, 1, kh, kw]
        new_conv.weight.copy_(averaged.repeat(1, in_channels, 1, 1))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class SegformerSegmentation(nn.Module):
    def __init__(self, hf_model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self._hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._hf_model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_segformer(
    variant: str = "b0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 5,
    classes: int = 1,
) -> nn.Module:
    """
    Build a SegFormer (B0 or B2) adapted to our 5-channel input /
    1-channel binary output. encoder_weights=None builds an untrained
    model of the same size from SegformerConfig instead of downloading
    the pretrained checkpoint -- for a from-scratch comparison, the same
    role encoder_weights=None plays in models/unet.py's build_unet.
    """
    if variant not in _CHECKPOINTS:
        raise ValueError(f"segformer variant {variant!r} isn't wired up -- expected one of b0, b2")

    if encoder_weights is not None:
        hf_model = SegformerForSemanticSegmentation.from_pretrained(_CHECKPOINTS[variant])
    else:
        config = SegformerConfig.from_pretrained(_CHECKPOINTS[variant])
        hf_model = SegformerForSemanticSegmentation(config)

    # Only the first stage's patch embedding reads the raw input; later
    # stages read the previous stage's output, so only this one conv
    # needs adapting.
    first_stage = hf_model.segformer.stages[0]
    first_stage.patch_embeddings.proj = _adapt_first_conv(
        first_stage.patch_embeddings.proj, in_channels
    )

    old_classifier = hf_model.decode_head.classifier
    hf_model.decode_head.classifier = nn.Conv2d(old_classifier.in_channels, classes, kernel_size=1)

    return SegformerSegmentation(hf_model)
