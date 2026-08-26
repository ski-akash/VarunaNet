"""
Tests for inference/streaming.py.

The important one is the equivalence test: streaming must produce exactly
what the in-memory path produces. The whole design rests on "rows above
the next tile row can never change again", and if that reasoning is off by
even one row the result is a horizontal seam every tile row -- which looks
plausible enough at a glance to ship unnoticed.
"""

import numpy as np
import pytest

from inference.streaming import (
    array_reader,
    collect_bands,
    predict_scene_streaming,
)
from inference.tiling import iter_tiles, predict_scene

TILE = 40
OVERLAP = 10
IN_CHANNELS = 5


class _PatternSession:
    """
    A stand-in ONNX session whose output depends on the input, so a tile
    read from the wrong place or written to the wrong place changes the
    result. A constant-output stub would pass even if the plumbing were
    scrambled.
    """

    def get_inputs(self):
        class _Spec:
            name = "input"

        return [_Spec()]

    def run(self, _outputs, feed):
        batch = feed["input"]
        # Mean across channels, scaled -- an arbitrary but input-dependent map.
        logits = batch.mean(axis=1, keepdims=True) * 3.0
        return [logits.astype(np.float32)]


def _scene(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(IN_CHANNELS, height, width)).astype(np.float32)


@pytest.mark.parametrize(
    "height,width",
    [
        (100, 130),  # neither dimension a multiple of the tile
        (80, 80),  # exact multiples
        (40, 40),  # exactly one tile
        (45, 200),  # very wide, barely taller than one tile
    ],
)
def test_streaming_matches_the_in_memory_path_exactly(height, width):
    scene = _scene(height, width)
    session = _PatternSession()

    reference = predict_scene(
        session, scene, tile_size=TILE, overlap=OVERLAP, batch_size=3
    )

    mask, writer = collect_bands(height, width)
    predict_scene_streaming(
        session,
        array_reader(scene),
        height,
        width,
        writer,
        tile_size=TILE,
        overlap=OVERLAP,
        batch_size=3,
    )

    assert np.array_equal(mask, reference)


def test_streaming_processes_every_tile():
    height, width = 100, 130
    session = _PatternSession()
    mask, writer = collect_bands(height, width)

    tiles_done = predict_scene_streaming(
        session,
        array_reader(_scene(height, width)),
        height,
        width,
        writer,
        tile_size=TILE,
        overlap=OVERLAP,
    )

    assert tiles_done == len(list(iter_tiles(height, width, TILE, OVERLAP)))


def test_every_row_is_written_exactly_once():
    """
    Bands must tile the scene: a gap leaves rows unpredicted, an overlap
    means a band was written twice and the second write may have been
    computed from incomplete accumulation.
    """
    height, width = 100, 130
    written_rows: list[int] = []

    def writer(first_row, band):
        written_rows.extend(range(first_row, first_row + band.shape[0]))

    predict_scene_streaming(
        _PatternSession(),
        array_reader(_scene(height, width)),
        height,
        width,
        writer,
        tile_size=TILE,
        overlap=OVERLAP,
    )

    assert sorted(written_rows) == list(range(height))


def test_bands_arrive_in_order_and_are_never_scene_sized():
    """
    A caller streaming to a GeoTIFF needs bands in row order, and the
    memory argument only holds if no single band is the whole scene.
    """
    height, width = 200, 130
    seen: list[tuple[int, int]] = []

    def writer(first_row, band):
        seen.append((first_row, band.shape[0]))

    predict_scene_streaming(
        _PatternSession(),
        array_reader(_scene(height, width)),
        height,
        width,
        writer,
        tile_size=TILE,
        overlap=OVERLAP,
    )

    assert [row for row, _ in seen] == sorted(row for row, _ in seen)
    assert len(seen) > 1, "a taller-than-one-tile scene should flush several bands"
    assert all(rows <= TILE for _, rows in seen)


def test_reader_is_only_ever_asked_for_one_tile_at_a_time():
    """
    The point of the reader callable is that the scene never has to be
    resident. If anything asked it for a scene-sized window that would be
    silently defeated.
    """
    height, width = 100, 130
    requested: list[tuple[int, int, int]] = []
    scene = _scene(height, width)

    def reader(row, col, size):
        requested.append((row, col, size))
        return scene[:, row : row + size, col : col + size]

    mask, writer = collect_bands(height, width)
    predict_scene_streaming(
        _PatternSession(), reader, height, width, writer, tile_size=TILE, overlap=OVERLAP
    )

    assert requested, "the reader was never called"
    assert all(size == TILE for _, _, size in requested)
    assert all(row + size <= height and col + size <= width for row, col, size in requested)


def test_scene_smaller_than_a_tile_is_rejected():
    mask, writer = collect_bands(10, 10)
    with pytest.raises(ValueError, match="smaller than one"):
        predict_scene_streaming(
            _PatternSession(),
            array_reader(_scene(10, 10)),
            10,
            10,
            writer,
            tile_size=TILE,
        )
