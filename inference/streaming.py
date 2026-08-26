"""
Runs a full scene through the model without ever holding the scene, or
its stitching buffers, in memory.

`inference/tiling.predict_scene` takes an in-memory array and accumulates
scene-sized float64 buffers. For a real 25,000 x 16,000 Sentinel-1 scene
that is 8 GB of input plus 6.4 GB of accumulators -- about 14.4 GB peak,
which is more than the serving container will have. It is fine for a
subset and unusable for the thing the service actually exists to do.

The fix is that tiles are laid out in rows, and a row of tiles can only
affect a bounded band of scene rows. Once the next row of tiles starts
below row R, no future tile can touch any row above R, so everything above
it is final: threshold it, hand it to the writer, and drop it. Only a
band of `tile_size` rows is ever live.

    tile row k     [========]                 <- accumulating
    tile row k+1        [========]            <- overlaps the band above
                   ^....^                     <- final once k+1 starts:
                                                 no later tile reaches here

That takes the accumulators from 6.4 GB to `tile_size * width * 16` bytes
-- about 131 MB at 512 x 16,000 -- and the input from 8 GB to one tile at
a time. Both become independent of scene height.

The reader is a callable rather than a rasterio dataset so this module
stays testable without a real GeoTIFF, and so a caller can back it with a
COG, a local file, or object storage without this code knowing.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from inference.tiling import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE_SIZE,
    _logit_of,
    _starts,
    taper_weights,
)

# (row, col, size) -> (channels, size, size) float32
ReadWindow = Callable[[int, int, int], np.ndarray]

# (first_row, mask_band) -> None, where mask_band is (rows, width) bool
WriteBand = Callable[[int, np.ndarray], None]


def predict_scene_streaming(
    session,
    read_window: ReadWindow,
    height: int,
    width: int,
    write_band: WriteBand,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> int:
    """
    Predict a whole scene band by band, writing finished rows as they go.

    Returns the number of tiles processed. The mask is delivered through
    `write_band` rather than returned, because a scene-sized bool array is
    400 MB and the whole point here is not to hold one.

    Identical output to tiling.predict_scene on the same input -- the
    blending maths is the same, just evaluated in a rolling window. That
    equivalence is asserted in the tests rather than assumed, since a
    subtle difference would show up as a horizontal seam every tile row.
    """
    if height < tile_size or width < tile_size:
        raise ValueError(
            f"scene {height}x{width} is smaller than one {tile_size}x{tile_size} tile"
        )

    stride = tile_size - overlap
    row_starts = _starts(height, tile_size, stride)
    col_starts = _starts(width, tile_size, stride)
    weight_map = taper_weights(tile_size, overlap)
    logit_threshold = _logit_of(threshold)
    input_name = session.get_inputs()[0].name

    # The live band: rows [band_start, band_start + tile_size).
    accumulated = np.zeros((tile_size, width), dtype=np.float64)
    weights = np.zeros((tile_size, width), dtype=np.float64)
    band_start = row_starts[0]
    tiles_done = 0

    for index, row in enumerate(row_starts):
        _shift_band_to(accumulated, weights, band_start, row)
        band_start = row

        for chunk in _chunked(col_starts, batch_size):
            batch = np.stack(
                [read_window(row, col, tile_size) for col in chunk]
            ).astype(np.float32)
            logits = session.run(None, {input_name: batch})[0]
            tiles_done += len(chunk)

            for offset, col in enumerate(chunk):
                patch = np.squeeze(logits[offset])
                local = slice(0, tile_size)
                accumulated[local, col : col + tile_size] += patch * weight_map
                weights[local, col : col + tile_size] += weight_map

        # Rows above the next tile row can no longer change: finalise them.
        # On the last row band that is everything remaining.
        final_until = row_starts[index + 1] if index + 1 < len(row_starts) else height
        _flush(
            accumulated,
            weights,
            band_start,
            final_until,
            logit_threshold,
            write_band,
        )

    return tiles_done


def _chunked(items: list[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _shift_band_to(
    accumulated: np.ndarray, weights: np.ndarray, old_start: int, new_start: int
) -> None:
    """
    Slide the live band down so it begins at `new_start`, in place.

    Rows that remain in range keep their partial accumulation -- they sit
    in the overlap between two tile rows and are still owed a contribution
    from the row about to be processed. Rows scrolling in from below start
    at zero. Done with a copy into the same buffers rather than
    reallocating, so peak memory stays flat across the whole scene.
    """
    shift = new_start - old_start
    if shift <= 0:
        return

    height = accumulated.shape[0]
    if shift >= height:
        accumulated[:] = 0.0
        weights[:] = 0.0
        return

    accumulated[:-shift] = accumulated[shift:]
    weights[:-shift] = weights[shift:]
    accumulated[-shift:] = 0.0
    weights[-shift:] = 0.0


def _flush(
    accumulated: np.ndarray,
    weights: np.ndarray,
    band_start: int,
    final_until: int,
    logit_threshold: float,
    write_band: WriteBand,
) -> None:
    """Threshold rows [band_start, final_until) and hand them to the writer."""
    rows = final_until - band_start
    if rows <= 0:
        return

    band_weights = weights[:rows]
    if np.any(band_weights == 0):
        raise RuntimeError(
            f"rows {band_start}..{final_until} were not covered by any tile -- "
            "the tile layout and the streaming band disagree about coverage"
        )

    blended = accumulated[:rows] / band_weights
    write_band(band_start, blended > logit_threshold)


def array_reader(scene: np.ndarray) -> ReadWindow:
    """
    A ReadWindow backed by an in-memory array, for tests and small scenes.

    Real callers pass a reader that pulls windows from a GeoTIFF/COG; this
    exists so the streaming logic can be tested against an exact reference
    without needing raster fixtures.
    """

    def read(row: int, col: int, size: int) -> np.ndarray:
        return scene[:, row : row + size, col : col + size]

    return read


def collect_bands(height: int, width: int) -> tuple[np.ndarray, WriteBand]:
    """
    A WriteBand that assembles the full mask in memory, plus the array it
    fills. Only for tests and small scenes -- assembling the whole mask is
    exactly what streaming exists to avoid.
    """
    mask = np.zeros((height, width), dtype=bool)

    def write(first_row: int, band: np.ndarray) -> None:
        mask[first_row : first_row + band.shape[0]] = band

    return mask, write
