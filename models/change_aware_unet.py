"""
ChangeAwareUNet -- an original two-stream fusion architecture, not from
any library, built for spec section 4.4's "stretch modeling work":
"Multi-temporal change detection... directly addresses permanent-water
confusion by learning change rather than absolute backscatter."

The spec's original framing assumes a true (pre-flood, during-flood) SAR
image pair, which Sen1Floods11 doesn't ship (each chip is a single
timestamp). The available proxy for "what water looked like before this
flood" is the JRC permanent-water mask (data/sen1floods11_normalization_stats
covers the 5 SAR/terrain channels, JRC is a separate binary layer --
see training/sen1floods11_dataset.py's include_jrc_baseline). So this
model takes two streams instead of one:

  - Main branch: the standard 5-channel input (VV_db, VH_db, VV_VH_ratio,
    slope, HAND), through a ResNet-34 encoder (ImageNet-pretrained, same
    channel-averaging adaptation as models/unet.py).
  - Baseline branch: the JRC permanent-water mask alone, through a small
    from-scratch CNN encoder sized to match the main branch's feature
    map resolutions/channel counts at each stage.

Fusing both streams' features (by concatenation) at every decoder skip
level -- not just feeding JRC in as an extra input channel -- lets the
decoder explicitly compare "what the SAR sees now" against "what's
normally wet" at every spatial scale, which is the actual mechanism a
change-aware model needs: a pixel that's SAR-wet AND JRC-wet is probably
permanent water, not flood; SAR-wet and JRC-dry is a much stronger flood
signal. A single extra input channel forces the network to rediscover
that relationship from scratch at every resolution; fusing it explicitly
in the skip connections hands it directly to the decoder instead.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Same pretrained-weight-averaging trick as models/unet.py's smp encoders."""
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
        averaged = conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(averaged.repeat(1, in_channels, 1, 1))
    return new_conv


class _BaselineEncoder(nn.Module):
    """
    Lightweight from-scratch CNN for the 1-channel JRC baseline branch --
    intentionally not a full pretrained ResNet: the input is a single
    binary mask, not a natural image, so ImageNet features wouldn't
    transfer, and the branch only needs to learn "where is permanent
    water" well enough to fuse with the main branch, not solve
    segmentation on its own. Output resolutions/channel counts are sized
    to match the main ResNet-34 branch's layer1-layer4 stages exactly,
    so concatenation at each skip level needs no interpolation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # -> H/4, 64ch (matches main branch's layer1)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # -> H/8, 128ch (matches layer2)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  # -> H/16, 256ch (matches layer3)
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )  # -> H/32, 512ch (matches layer4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b1 = self.stem(x)
        b2 = self.stage2(b1)
        b3 = self.stage3(b2)
        b4 = self.stage4(b3)
        return b1, b2, b3, b4


class _UpBlock(nn.Module):
    """Upsample-by-2, concat a fused skip (or nothing), then a double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _double_conv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ChangeAwareUNet(nn.Module):
    def __init__(self, main_in_channels: int = 5, classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet34(weights=weights)
        resnet.conv1 = _adapt_first_conv(resnet.conv1, main_in_channels)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # H/4,  64ch
        self.layer2 = resnet.layer2  # H/8,  128ch
        self.layer3 = resnet.layer3  # H/16, 256ch
        self.layer4 = resnet.layer4  # H/32, 512ch

        self.baseline_encoder = _BaselineEncoder()

        # Decoder consumes fused (main + baseline) skip connections, so
        # every skip_channels figure below is 2x the main branch's own
        # channel count at that stage.
        self.up4 = _UpBlock(in_channels=512 + 512, skip_channels=256 + 256, out_channels=256)
        self.up3 = _UpBlock(in_channels=256, skip_channels=128 + 128, out_channels=128)
        self.up2 = _UpBlock(in_channels=128, skip_channels=64 + 64, out_channels=64)
        # No skip available below H/4 (ResNet's stem already downsamples
        # that far before layer1) -- these two final stages just refine
        # the upsampled features back to full resolution.
        self.up1 = _UpBlock(in_channels=64, skip_channels=0, out_channels=32)
        self.up0 = _UpBlock(in_channels=32, skip_channels=0, out_channels=16)

        self.classifier = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_x, baseline_x = x[:, :-1], x[:, -1:]

        stem_out = self.stem(main_x)
        e1 = self.layer1(stem_out)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        b1, b2, b3, b4 = self.baseline_encoder(baseline_x)

        bottleneck = torch.cat([e4, b4], dim=1)
        skip3 = torch.cat([e3, b3], dim=1)
        skip2 = torch.cat([e2, b2], dim=1)
        skip1 = torch.cat([e1, b1], dim=1)

        d = self.up4(bottleneck, skip3)
        d = self.up3(d, skip2)
        d = self.up2(d, skip1)
        d = self.up1(d)
        d = self.up0(d)
        return self.classifier(d)


def build_change_aware_unet(
    encoder_weights: str | None = "imagenet",
    in_channels: int = 6,
    classes: int = 1,
) -> nn.Module:
    """
    in_channels=6 is the full input (5 SAR/terrain + 1 JRC baseline,
    see training/sen1floods11_dataset.py's include_jrc_baseline) --
    the model splits the last channel off internally as the baseline
    branch's input (see ChangeAwareUNet.forward).
    """
    return ChangeAwareUNet(
        main_in_channels=in_channels - 1,
        classes=classes,
        pretrained=encoder_weights is not None,
    )
