"""
Splits a full Sentinel-1 scene into model-sized tiles, runs them through
an ONNX session in batches, and stitches the predictions back into one
scene-sized mask.

A Sentinel-1 GRD scene is roughly 25,000 x 16,000 pixels; the model takes
512 x 512. So inference on a real scene is not one forward pass but about
1,500 of them, and the interesting problems are all at the joins.

**Why tiles overlap.** A convolutional segmentation model has far less
context at a tile's edge than at its centre -- the receptive field runs
off the end and gets padding instead of neighbouring water. Predictions
are therefore systematically worse near tile borders, and butt-jointed
tiles put those weak edges directly against each other, producing a
visible grid of seams across the flood mask. Overlapping the tiles means
every pixel near one tile's weak edge sits near another tile's strong
centre.

**Why a tapered blend rather than an average.** Averaging overlapping
predictions equally still lets a weak edge pull a good centre around, and
leaves a faint discontinuity where the overlap region starts and stops.
Weighting each tile's contribution by a taper that peaks at its centre and
falls to (almost) zero at its border makes the transition continuous: at
any pixel the tile that can see the most context around it dominates. The
weights are accumulated alongside the predictions and divided out at the
end, so the result is a proper weighted mean, not a sum.

Blending happens on **logits, not thresholded masks**. Averaging binary
decisions throws away all confidence information and produces ragged
half-committed boundaries; averaging logits then thresholding once at the
end is the same reasoning behind logit-averaged ensembling in
training/evaluate_ensemble.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Matches the tile size the model is trained on and export_onnx.py pins
# into the graph.
DEFAULT_TILE_SIZE = 512

# How much neighbouring tiles overlap, in pixels. A quarter of the tile is
# enough that every pixel is within some tile's strong central region,
# without inflating the number of forward passes too far: at 512/128 a
# scene needs ~1.8x the tiles of a non-overlapping grid.
DEFAULT_OVERLAP = 128


@dataclass(frozen=True)
class Tile:
    """One tile's placement in the scene. `row`/`col` are top-left."""

    row: int
    col: int
    size: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.size)

    @property
    def col_slice(self) -> slice:
        return slice(self.col, self.col + self.size)


def _starts(extent: int, tile_size: int, stride: int) -> list[int]:
    """
    Tile start offsets along one axis, always ending flush with the scene.

    The final start is clamped so the last tile ends exactly at `extent`
    rather than running past it. That makes the last step shorter than the
    others (tiles overlap more there), which is harmless -- the blend
    handles uneven overlap by construction -- and avoids either padding the
    scene or leaving a strip along the bottom and right edges unpredicted.
    A scene's dimensions are essentially never an exact multiple of the
    tile size, so this is the normal case, not an edge case.
    """
    if extent <= tile_size:
        return [0]

    starts = list(range(0, extent - tile_size + 1, stride))
    if starts[-1] != extent - tile_size:
        starts.append(extent - tile_size)
    return starts


def iter_tiles(
    height: int,
    width: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[Tile]:
    """Yield every tile position covering a `height` x `width` scene."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")

    stride = tile_size - overlap
    for row in _starts(height, tile_size, stride):
        for col in _starts(width, tile_size, stride):
            yield Tile(row=row, col=col, size=tile_size)


def taper_weights(tile_size: int, overlap: int) -> np.ndarray:
    """
    A 2D weight map peaking at the tile centre and falling off at the edges.

    Built as the outer product of a 1D ramp so it is separable and cheap.
    The ramp rises over the overlap region and is flat across the middle,
    which means a pixel in the flat centre of one tile always outweighs the
    same pixel sitting in another tile's ramp.

    Never exactly zero (floored at a small epsilon): a pixel covered by
    only one tile -- which happens along the scene's outer border, where
    there is no neighbour to overlap with -- must still end up with a
    non-zero total weight, or dividing by that total produces NaN.
    """
    if overlap == 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    ramp = np.ones(tile_size, dtype=np.float32)
    edge = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    ramp = np.maximum(ramp, 1e-6)
    return np.outer(ramp, ramp).astype(np.float32)


def stitch_logits(
    tile_logits: list[np.ndarray],
    tiles: list[Tile],
    height: int,
    width: int,
    overlap: int = DEFAULT_OVERLAP,
) -> np.ndarray:
    """
    Combine per-tile logits into one `height` x `width` logit map.

    Accumulates weighted logits and the weights themselves, then divides,
    giving a weighted mean rather than a sum. float64 accumulators: a large
    scene sums many float32 contributions per pixel and the rounding is
    otherwise visible against the threshold.
    """
    if len(tile_logits) != len(tiles):
        raise ValueError(
            f"got {len(tile_logits)} tile predictions for {len(tiles)} tiles"
        )
    if not tiles:
        raise ValueError("no tiles to stitch")

    accumulated = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    weight_map = taper_weights(tiles[0].size, overlap)

    for logits, tile in zip(tile_logits, tiles):
        patch = np.squeeze(logits)
        if patch.shape != (tile.size, tile.size):
            raise ValueError(
                f"tile prediction shape {patch.shape} != ({tile.size}, {tile.size})"
            )
        accumulated[tile.row_slice, tile.col_slice] += patch * weight_map
        weights[tile.row_slice, tile.col_slice] += weight_map

    if np.any(weights == 0):
        raise RuntimeError(
            "some pixels were not covered by any tile -- iter_tiles and "
            "stitch_logits disagree about scene coverage"
        )
    return (accumulated / weights).astype(np.float32)


def predict_scene(
    session,
    scene: np.ndarray,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run a whole scene through an ONNX session and return a boolean water mask.

    `scene` is (channels, height, width) -- the same channel-first layout
    the data contract and the model use.

    Tiles are pushed through in batches because per-tile calls waste most
    of the runtime's throughput on ~1,500 separate invocations. The batch
    axis is dynamic in the exported graph precisely so this can be tuned to
    whatever machine is serving (see inference/export_onnx.py).
    """
    if scene.ndim != 3:
        raise ValueError(f"expected (channels, height, width), got shape {scene.shape}")

    _, height, width = scene.shape
    if height < tile_size or width < tile_size:
        raise ValueError(
            f"scene {height}x{width} is smaller than one {tile_size}x{tile_size} tile"
        )

    tiles = list(iter_tiles(height, width, tile_size, overlap))
    input_name = session.get_inputs()[0].name

    predictions: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        batch = np.stack(
            [scene[:, t.row_slice, t.col_slice] for t in chunk]
        ).astype(np.float32)
        logits = session.run(None, {input_name: batch})[0]
        predictions.extend(logits[i] for i in range(len(chunk)))

    stitched = stitch_logits(predictions, tiles, height, width, overlap)
    # Threshold once, at the end, on blended logits -- see module docstring.
    return stitched > _logit_of(threshold)


def _logit_of(probability: float) -> float:
    """
    The logit corresponding to a probability threshold.

    Thresholding blended logits directly avoids a sigmoid over the whole
    scene: for the default 0.5 this is simply 0.0, and sigmoid is monotonic
    so comparing logits against logit(p) is identical to comparing
    sigmoid(logits) against p.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {probability}")
    return float(np.log(probability / (1.0 - probability)))
