"""
Single entry point for building any of this project's architectures
(spec section 4.3's Phase 4 architecture study), so every caller
(training/train.py, training/evaluate_test.py, training/per_event_eval.py)
picks a model the same way regardless of which family it comes from --
segmentation_models_pytorch (U-Net, U-Net++, DeepLabV3+; see
models/unet.py) or HuggingFace's SegFormer (models/segformer.py).

Kept separate from models/unet.py itself so that module's docstring
("we don't hand-roll U-Net... same architecture family... all through
one consistent interface") stays accurate -- it covers the smp family,
not SegFormer's fundamentally different transformer/HF interface.
"""

from torch import nn

from models.segformer import build_segformer
from models.unet import _ARCHITECTURES, build_unet

_SEGFORMER_VARIANTS = {"segformer_b0": "b0", "segformer_b2": "b2"}


def build_model(
    architecture: str,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 5,
    classes: int = 1,
) -> nn.Module:
    if architecture in _ARCHITECTURES:
        return build_unet(
            architecture=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if architecture in _SEGFORMER_VARIANTS:
        return build_segformer(
            variant=_SEGFORMER_VARIANTS[architecture],
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    raise ValueError(
        f"architecture {architecture!r} isn't wired up yet -- expected one of "
        f"{sorted(_ARCHITECTURES) + sorted(_SEGFORMER_VARIANTS)}"
    )
