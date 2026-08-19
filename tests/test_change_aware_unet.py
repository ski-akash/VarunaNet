"""
Tests for models/change_aware_unet.py.

Same discipline as tests/test_unet.py (spec section 8: exercise the model
wiring on tiny synthetic tensors before any real GPU job runs) -- this file
existed for build_unet but not yet for ChangeAwareUNet, even though
training/train_change_aware_p100.sh is about to submit it to a real cluster
job. pretrained=False everywhere except the one test that checks the
default, to avoid downloading ImageNet weights over the network on every
test run.
"""

import inspect

import torch

from models.build_model import build_model
from models.change_aware_unet import ChangeAwareUNet, build_change_aware_unet

# Smallest size divisible by 32 (ResNet's total encoder downsampling
# factor), matching test_unet.py's TINY_SIZE for the same reason.
TINY_SIZE = 64
FULL_IN_CHANNELS = 6  # 5 SAR/terrain + 1 JRC baseline, per build_change_aware_unet's docstring


def test_default_pretrained_is_true():
    default = inspect.signature(build_change_aware_unet).parameters["encoder_weights"].default
    assert default == "imagenet"


def test_default_in_channels_is_six():
    default = inspect.signature(build_change_aware_unet).parameters["in_channels"].default
    assert default == FULL_IN_CHANNELS


def test_forward_pass_shape():
    model = build_change_aware_unet(encoder_weights=None, in_channels=FULL_IN_CHANNELS)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, FULL_IN_CHANNELS, TINY_SIZE, TINY_SIZE)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, 1, TINY_SIZE, TINY_SIZE)


def test_output_is_raw_logits_not_probabilities():
    # Checked structurally rather than by pushing input to extreme values
    # and hoping the output lands outside [0, 1]: this model's decoder is
    # BatchNorm-heavy, which renormalizes activations at every stage
    # regardless of input scale, so that trick (used in test_unet.py,
    # where it works) isn't reliable here -- an untrained ChangeAwareUNet
    # can land its raw-logit output inside [0, 1] by pure chance on a
    # given random init, which isn't itself a bug. What actually matters
    # -- no sigmoid baked into forward() -- is a structural fact about the
    # model, so assert that directly instead.
    model = build_change_aware_unet(encoder_weights=None, in_channels=FULL_IN_CHANNELS)
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())
    assert isinstance(model.classifier, torch.nn.Conv2d)


def test_gradients_flow_to_both_branches():
    # Regression check for the two-stream fusion: if either branch were
    # accidentally detached (e.g. the baseline encoder's output not
    # actually reaching the decoder), gradients would silently stop
    # flowing to that branch's input channels and it would never learn.
    model = build_change_aware_unet(encoder_weights=None, in_channels=FULL_IN_CHANNELS)
    model.train()

    x = torch.randn(1, FULL_IN_CHANNELS, TINY_SIZE, TINY_SIZE, requires_grad=True)
    logits = model(x)
    logits.sum().backward()

    assert x.grad is not None
    main_grad, baseline_grad = x.grad[:, :-1], x.grad[:, -1:]
    assert torch.any(main_grad != 0)
    assert torch.any(baseline_grad != 0)


def test_last_channel_is_the_baseline_branch_input():
    # Regression check for build_change_aware_unet's channel split
    # (main_x, baseline_x = x[:, :-1], x[:, -1:]) -- confirms the JRC
    # baseline channel really is read from the *last* input channel, not
    # silently misaligned with how training/sen1floods11_dataset.py's
    # include_jrc_baseline appends it.
    model = ChangeAwareUNet(main_in_channels=FULL_IN_CHANNELS - 1, pretrained=False)
    model.eval()

    x_zero_baseline = torch.randn(1, FULL_IN_CHANNELS, TINY_SIZE, TINY_SIZE)
    x_zero_baseline[:, -1:] = 0.0
    x_nonzero_baseline = x_zero_baseline.clone()
    x_nonzero_baseline[:, -1:] = 1.0

    with torch.no_grad():
        out_zero = model(x_zero_baseline)
        out_nonzero = model(x_nonzero_baseline)

    assert not torch.allclose(out_zero, out_nonzero)


def test_build_model_dispatches_to_change_aware_unet():
    model = build_model(
        architecture="change_aware_unet", encoder_weights=None, in_channels=FULL_IN_CHANNELS
    )
    assert isinstance(model, ChangeAwareUNet)
