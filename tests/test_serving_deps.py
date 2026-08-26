"""
Guards the property the whole serving architecture rests on: importing the
inference service must not drag in the training stack.

infra/Dockerfile.serve is slim only because `inference.service` and
everything it imports avoid torch, torchvision,
segmentation-models-pytorch and transformers -- 647MB of installed
packages against 211MB for the serving set, before any CUDA layers.

This is exactly the kind of property that regresses silently. One
convenient `from models.build_model import build_model` at the top of a
serving module -- and that import is genuinely tempting, since
export_onnx.py legitimately needs it -- would pull the whole tree back in,
and nothing would fail. The image would just quietly get 3x bigger, which
nobody notices until a deploy.

The check runs in a subprocess because imports are process-global: pytest
has almost certainly already imported torch for the training tests, so
asking `sys.modules` in-process would always report it present.
"""

import subprocess
import sys

# Packages that belong to training and must never appear in a serving
# process. transformers arrives via SegFormer, smp via the U-Net family.
TRAINING_ONLY = ("torch", "torchvision", "segmentation_models_pytorch", "transformers")

# Modules that must stay importable without the training stack. Anything
# the container's entrypoint touches belongs here.
SERVING_MODULES = (
    "inference.service",
    "inference.pipeline",
    "inference.tiling",
    "inference.streaming",
    "inference.vectorize",
    "inference.districts",
)


def _imports_after(module: str) -> set[str]:
    """Which TRAINING_ONLY packages are loaded after importing `module`."""
    code = (
        "import sys\n"
        f"import {module}\n"
        f"print(','.join(m for m in {TRAINING_ONLY!r} if m in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return {name for name in completed.stdout.strip().split(",") if name}


def test_the_service_entrypoint_does_not_import_the_training_stack():
    leaked = _imports_after("inference.service")

    assert not leaked, (
        f"inference.service pulled in {sorted(leaked)}. The serving image is slim "
        "only because it does not contain the training stack (647MB vs 211MB of "
        "installed packages); a convenience import here silently triples it."
    )


def test_no_serving_module_imports_the_training_stack():
    offenders = {
        module: sorted(leaked)
        for module in SERVING_MODULES
        if (leaked := _imports_after(module))
    }

    assert not offenders, f"training packages leaked into serving modules: {offenders}"


def test_export_is_allowed_to_need_torch():
    """
    The counterpart, stated so the boundary is explicit rather than
    implied: export_onnx.py runs at build time on the training side and
    legitimately needs torch. The rule is about what the *serving* process
    loads, not a blanket ban.
    """
    assert _imports_after("inference.export_onnx") & {"torch"}
