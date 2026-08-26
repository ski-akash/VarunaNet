"""
Quantizes an exported ONNX model and measures what that actually costs.

Spec section 5 asks for "a real quantization/latency story (FP32 vs FP16
vs INT8)". The point of this module is that it must be *measured*, both
halves of it: quantization is only worth shipping if the speedup is real
AND the predictions barely move, and the second half is the one that gets
skipped.

**Benchmark on real chips, not random noise.** Both were measured on the
same two models and they disagree in *both* directions:

    random noise : 2.80x faster, 0.135% pixels flipped, max logit shift 0.297
    16 real chips: 1.71x faster, 0.258% pixels flipped, max logit shift 2.521

Random Gaussian input has none of SAR's dynamic range or spatial
structure, and activation ranges are exactly what dynamic quantization
keys off. So it understates the accuracy cost (by ~2x) and overstates the
speedup (2.80x against a real 1.71x) -- it would have supported a
confident claim that is wrong twice over. The CLI loads real chips when
the dataset is present and says so loudly when it falls back.

So `benchmark_quantization` reports latency and agreement together. The
accuracy metric is deliberately **disagreement on the thresholded mask**,
not mean squared error on logits: what ships is a water/not-water
decision, and a logit that shifts from 4.1 to 4.0 changes nothing while
one that crosses zero flips a pixel. MSE averages those two cases
together and hides the only one that matters.

Dynamic INT8 quantization is used rather than static. Static quantization
gives better accuracy but needs a calibration dataset -- real chips run
through the model to observe activation ranges -- which makes it a
data-dependent build step. Dynamic quantization computes activation
ranges at inference time instead: no calibration data, no extra build
input, at some accuracy cost. Whether that cost is acceptable is exactly
what this module measures rather than assumes.

Run directly:
    python -m inference.quantize --model model.onnx --output model.int8.onnx
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Tiles to time. Latency on the first run includes graph optimisation and
# memory arena setup, which is not what serving throughput looks like, so
# a warmup run is always discarded before timing.
DEFAULT_WARMUP_RUNS = 2
DEFAULT_TIMED_RUNS = 5


@dataclass
class QuantizationReport:
    fp32_path: Path
    int8_path: Path
    fp32_bytes: int
    int8_bytes: int
    fp32_seconds_per_tile: float
    int8_seconds_per_tile: float
    mask_disagreement_fraction: float
    max_abs_logit_diff: float

    @property
    def size_ratio(self) -> float:
        return self.fp32_bytes / self.int8_bytes if self.int8_bytes else float("nan")

    @property
    def speedup(self) -> float:
        return (
            self.fp32_seconds_per_tile / self.int8_seconds_per_tile
            if self.int8_seconds_per_tile
            else float("nan")
        )


def quantize_dynamic_int8(model_path: Path, output_path: Path) -> Path:
    """
    Write a dynamically-quantized INT8 copy of `model_path`.

    Weights become int8; activations are quantized on the fly using ranges
    computed per inference. QUInt8 for weights rather than QInt8 because
    ONNX Runtime's CPU kernels are better optimised for the unsigned form
    on the convolution-heavy graphs this project exports.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )
    return output_path


def _time_session(session, batch: np.ndarray, warmup: int, timed: int) -> float:
    """Median seconds per tile, warmup runs discarded."""
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {input_name: batch})

    durations = []
    for _ in range(timed):
        start = time.perf_counter()
        session.run(None, {input_name: batch})
        durations.append(time.perf_counter() - start)

    # Median, not mean: a single scheduler hiccup on a shared machine
    # skews a mean of five runs badly, and the typical case is what
    # serving throughput depends on.
    return float(np.median(durations)) / batch.shape[0]


def load_real_chips(count: int, data_root: Path = Path("datasets/sen1floods11")) -> np.ndarray | None:
    """
    A batch of real Sen1Floods11 test chips, or None if the dataset isn't
    present (it is gitignored and large, so this must degrade gracefully).
    """
    if not (data_root / "splits" / "flood_test_data.csv").exists():
        return None

    from data.normalization import NormalizationStats
    from training.sen1floods11_dataset import Sen1Floods11TorchDataset

    stats = NormalizationStats.load("data/sen1floods11_normalization_stats.json")
    dataset = Sen1Floods11TorchDataset(
        image_dir=data_root / "S1Hand",
        label_dir=data_root / "LabelHand",
        dem_dir=data_root / "DEMHand",
        split_csv=data_root / "splits" / "flood_test_data.csv",
        normalization_stats=stats,
    )
    count = min(count, len(dataset))
    return np.stack([dataset[i][0].numpy() for i in range(count)]).astype(np.float32)


def benchmark_quantization(
    fp32_path: Path,
    int8_path: Path,
    in_channels: int = 5,
    tile_size: int = 512,
    batch_size: int = 2,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    timed_runs: int = DEFAULT_TIMED_RUNS,
    seed: int = 0,
    sample_batch: np.ndarray | None = None,
) -> QuantizationReport:
    """
    Compare an FP32 model against its INT8 counterpart on the same input.

    Both models see byte-identical input, so any difference in output is
    quantization and nothing else.

    `sample_batch` should be real chips wherever possible -- see the module
    docstring on why random input understates the accuracy cost. When it is
    None a random batch is generated, which is fine for testing the
    plumbing and misleading for reporting a result.
    """
    import onnxruntime

    if sample_batch is None:
        rng = np.random.default_rng(seed)
        batch = rng.normal(
            size=(batch_size, in_channels, tile_size, tile_size)
        ).astype(np.float32)
    else:
        batch = sample_batch.astype(np.float32)

    fp32 = onnxruntime.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    int8 = onnxruntime.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    )

    fp32_logits = fp32.run(None, {fp32.get_inputs()[0].name: batch})[0]
    int8_logits = int8.run(None, {int8.get_inputs()[0].name: batch})[0]

    # The metric that matters: how many pixels actually flip their
    # water/not-water decision. Compared at logit 0, which is sigmoid 0.5.
    disagreement = float(np.mean((fp32_logits > 0) != (int8_logits > 0)))
    max_abs_diff = float(np.max(np.abs(fp32_logits - int8_logits)))

    return QuantizationReport(
        fp32_path=fp32_path,
        int8_path=int8_path,
        fp32_bytes=fp32_path.stat().st_size,
        int8_bytes=int8_path.stat().st_size,
        fp32_seconds_per_tile=_time_session(fp32, batch, warmup_runs, timed_runs),
        int8_seconds_per_tile=_time_session(int8, batch, warmup_runs, timed_runs),
        mask_disagreement_fraction=disagreement,
        max_abs_logit_diff=max_abs_diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="FP32 .onnx input")
    parser.add_argument("--output", required=True, type=Path, help="INT8 .onnx output")
    parser.add_argument("--in-channels", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--chips",
        type=int,
        default=16,
        help="real test chips to benchmark on (0 to force random input)",
    )
    args = parser.parse_args()

    quantize_dynamic_int8(args.model, args.output)

    sample = load_real_chips(args.chips) if args.chips > 0 else None
    if sample is None:
        print(
            "WARNING: benchmarking on random input -- the Sen1Floods11 test split was "
            "not found. Random noise understates the accuracy cost of quantization "
            "(measured ~2x lower than real chips); treat the agreement figures below "
            "as a lower bound, not a result."
        )
    else:
        print(f"benchmarking on {sample.shape[0]} real test chips")

    report = benchmark_quantization(
        args.model,
        args.output,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        sample_batch=sample,
    )

    print(
        f"FP32 {report.fp32_bytes / 1e6:7.1f} MB  "
        f"{report.fp32_seconds_per_tile * 1000:7.1f} ms/tile\n"
        f"INT8 {report.int8_bytes / 1e6:7.1f} MB  "
        f"{report.int8_seconds_per_tile * 1000:7.1f} ms/tile\n"
        f"  {report.size_ratio:.2f}x smaller, {report.speedup:.2f}x faster\n"
        f"  mask pixels changed: {report.mask_disagreement_fraction * 100:.4f}%\n"
        f"  max abs logit shift: {report.max_abs_logit_diff:.3f}"
    )


if __name__ == "__main__":
    main()
