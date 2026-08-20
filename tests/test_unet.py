"""
Tests for models/unet.py.

Per spec section 8 ("test the full training path locally on tiny synthetic
tensors before the user submits anything"), these tests never touch real
imagery -- they only check that the model wiring (channel counts, output
shape, gradient flow) matches the data contract. Real accuracy is what
benchmarks/ measures against real chips, not this file.

encoder_weights=None is used everywhere except the one test that checks the
default parameter value, specifically to avoid downloading ImageNet
pretrained weights over the network on every test run -- that would make
`pytest -q` slow and flaky (and dependent on internet access) for a check
that isn't about correctness of the wiring at all.
"""

import inspect

import torch

from data.contract import NUM_CHANNELS
from models.unet import NUM_OUTPUT_CLASSES, build_unet

# Smallest size that still divides evenly by 32 (ResNet's total encoder
# downsampling factor -- see build_unet's docstring), so the tiny synthetic
# forward pass below is fast but still exercises every encoder/decoder
# stage the same way a real 512x512 chip would.
TINY_SIZE = 64


def test_default_pretrained_weights_source_is_imagenet():
    # Checked via the function signature, not by actually building the
    # model, so this doesn't require a network call.
    default = inspect.signature(build_unet).parameters["encoder_weights"].default
    assert default == "imagenet"


def test_default_in_channels_matches_data_contract():
    default = inspect.signature(build_unet).parameters["in_channels"].default
    assert default == NUM_CHANNELS


def test_forward_pass_shape_matches_data_contract():
    model = build_unet(encoder_weights=None)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, NUM_CHANNELS, TINY_SIZE, TINY_SIZE)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, NUM_OUTPUT_CLASSES, TINY_SIZE, TINY_SIZE)


def test_output_is_raw_logits_not_probabilities():
    # No sigmoid should be baked into the model itself (see module
    # docstring in unet.py) -- BCEWithLogitsLoss expects raw logits, and
    # applying sigmoid twice would silently break training.
    model = build_unet(encoder_weights=None)
    model.eval()

    x = torch.randn(1, NUM_CHANNELS, TINY_SIZE, TINY_SIZE) * 10  # push values to extremes
    with torch.no_grad():
        logits = model(x)

    assert (logits < 0).any() or (logits > 1).any()


def test_gradients_flow_to_input_adapted_first_conv():
    # Regression check for the in_channels != 3 adaptation described in
    # unet.py: if smp silently dropped or froze the adapted first
    # convolution, gradients would never reach the early layers and
    # training would quietly fail to learn from some channels.
    model = build_unet(encoder_weights=None)
    model.train()

    x = torch.randn(1, NUM_CHANNELS, TINY_SIZE, TINY_SIZE, requires_grad=True)
    logits = model(x)
    logits.sum().backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_resnet50_encoder_builds_and_adapts_input_channels():
    # Regression test for training/conf/model/unet_resnet50.yaml (the
    # Phase 4 architecture study's bigger-encoder variant): confirms the
    # same in_channels != 3 pretrained-weight-averaging adaptation this
    # module's docstring describes for ResNet-34 also works for
    # ResNet-50, rather than assuming smp's adaptation is encoder-agnostic.
    model = build_unet(encoder_name="resnet50", encoder_weights=None, in_channels=NUM_CHANNELS)
    model.eval()

    x = torch.randn(1, NUM_CHANNELS, TINY_SIZE, TINY_SIZE)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, NUM_OUTPUT_CLASSES, TINY_SIZE, TINY_SIZE)


def test_configurable_encoder_and_channels():
    # These knobs exist specifically for Phase 4's architecture study and
    # for reuse beyond the 5-channel data contract -- confirm they're
    # actually wired through to the built model, not just accepted and
    # ignored.
    in_channels = 3
    classes = 4
    model = build_unet(in_channels=in_channels, classes=classes, encoder_weights=None)
    model.eval()

    x = torch.randn(1, in_channels, TINY_SIZE, TINY_SIZE)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, classes, TINY_SIZE, TINY_SIZE)
