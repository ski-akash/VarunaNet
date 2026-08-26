"""
Tests for inference/tiling.py.

Small tile sizes throughout so the arrays stay readable; the properties
being checked (full coverage, exact reconstruction, no seams, correct
handling of scenes that aren't a multiple of the tile size) are the same
ones that matter at 512 on a 25,000 x 16,000 scene.
"""

import numpy as np
import pytest

from inference.tiling import (
    Tile,
    _logit_of,
    iter_tiles,
    predict_scene,
    stitch_logits,
    taper_weights,
)


def test_tiles_cover_every_pixel_of_the_scene():
    height, width, tile, overlap = 100, 130, 40, 10
    covered = np.zeros((height, width), dtype=int)

    for t in iter_tiles(height, width, tile, overlap):
        covered[t.row_slice, t.col_slice] += 1

    assert covered.min() >= 1, "a pixel was left unpredicted"


def test_last_tile_is_flush_with_the_scene_edge():
    """
    Scene dimensions are essentially never an exact multiple of the tile
    size. The final tile must be pulled back to end exactly at the edge,
    rather than overrunning it or leaving an unpredicted strip.
    """
    height, width, tile, overlap = 100, 130, 40, 10
    tiles = list(iter_tiles(height, width, tile, overlap))

    assert max(t.row + t.size for t in tiles) == height
    assert max(t.col + t.size for t in tiles) == width
    assert all(t.row >= 0 and t.col >= 0 for t in tiles)


def test_scene_smaller_than_one_tile_yields_a_single_tile():
    assert list(iter_tiles(10, 10, 40, 10)) == [Tile(0, 0, 40)]


@pytest.mark.parametrize("overlap", [0, 8, 39])
def test_valid_overlaps_are_accepted(overlap):
    assert list(iter_tiles(100, 100, 40, overlap))


@pytest.mark.parametrize("overlap", [-1, 40, 41])
def test_overlap_outside_the_tile_is_rejected(overlap):
    with pytest.raises(ValueError, match="overlap"):
        list(iter_tiles(100, 100, 40, overlap))


def test_taper_peaks_in_the_centre_and_falls_off_at_the_edges():
    weights = taper_weights(tile_size=40, overlap=10)

    assert weights.shape == (40, 40)
    assert weights[20, 20] == pytest.approx(1.0)
    # The ramp spans indices 0..overlap-1; index `overlap` is the first
    # flat value, so probe inside the ramp rather than at its far end.
    assert weights[0, 20] < weights[5, 20] < weights[9, 20]
    assert weights[10, 20] == pytest.approx(1.0)  # flat centre begins here
    # Never exactly zero: a border pixel covered by only one tile still
    # needs a non-zero total weight, or the divide produces NaN.
    assert weights.min() > 0


def test_stitching_reconstructs_a_known_field_exactly():
    """
    A weighted mean of identical values must return that value, whatever
    the weights are. Feeding each tile the true field means any error in
    the accumulate/normalise arithmetic shows up as a mismatch.
    """
    height, width, tile, overlap = 100, 130, 40, 10
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(height, width)).astype(np.float32)

    tiles = list(iter_tiles(height, width, tile, overlap))
    patches = [truth[t.row_slice, t.col_slice] for t in tiles]

    stitched = stitch_logits(patches, tiles, height, width, overlap)

    assert np.allclose(stitched, truth, atol=1e-5)


def test_stitching_a_constant_field_has_no_seams():
    """
    The failure this guards is visible grid lines across the flood mask
    where tile borders meet. A constant input must come back exactly
    constant -- any seam shows up as variance.
    """
    height, width, tile, overlap = 100, 130, 40, 10
    tiles = list(iter_tiles(height, width, tile, overlap))
    patches = [np.full((tile, tile), 3.5, dtype=np.float32) for _ in tiles]

    stitched = stitch_logits(patches, tiles, height, width, overlap)

    assert np.allclose(stitched, 3.5, atol=1e-5)
    assert stitched.std() < 1e-6


def test_stitching_rejects_mismatched_prediction_counts():
    tiles = list(iter_tiles(100, 100, 40, 10))
    with pytest.raises(ValueError, match="tile predictions"):
        stitch_logits([np.zeros((40, 40), dtype=np.float32)], tiles, 100, 100, 10)


def test_logit_of_default_threshold_is_zero():
    assert _logit_of(0.5) == pytest.approx(0.0)
    assert _logit_of(0.75) > 0
    assert _logit_of(0.25) < 0


class _ConstantSession:
    """
    Stands in for an onnxruntime InferenceSession, returning a fixed logit
    for every pixel so predict_scene's plumbing can be tested without a
    model. Records the batch shapes it saw.
    """

    def __init__(self, logit_value: float):
        self.logit_value = logit_value
        self.seen_batches: list[tuple] = []

    def get_inputs(self):
        class _Spec:
            name = "input"

        return [_Spec()]

    def run(self, _outputs, feed):
        batch = feed["input"]
        self.seen_batches.append(batch.shape)
        n = batch.shape[0]
        tile = batch.shape[-1]
        return [np.full((n, 1, tile, tile), self.logit_value, dtype=np.float32)]


def test_predict_scene_thresholds_blended_logits():
    session = _ConstantSession(logit_value=2.0)  # sigmoid(2) ~ 0.88 -> water
    scene = np.zeros((5, 100, 130), dtype=np.float32)

    mask = predict_scene(session, scene, tile_size=40, overlap=10, batch_size=4)

    assert mask.shape == (100, 130)
    assert mask.dtype == bool
    assert mask.all()


def test_predict_scene_respects_a_negative_logit():
    session = _ConstantSession(logit_value=-2.0)
    scene = np.zeros((5, 100, 130), dtype=np.float32)

    mask = predict_scene(session, scene, tile_size=40, overlap=10, batch_size=4)

    assert not mask.any()


def test_predict_scene_batches_tiles_rather_than_calling_per_tile():
    """
    ~1,500 single-tile invocations waste most of the runtime's throughput,
    which is the reason the exported graph keeps a dynamic batch axis.
    """
    session = _ConstantSession(logit_value=1.0)
    scene = np.zeros((5, 100, 130), dtype=np.float32)

    predict_scene(session, scene, tile_size=40, overlap=10, batch_size=4)

    assert max(shape[0] for shape in session.seen_batches) > 1
    assert all(shape[1] == 5 for shape in session.seen_batches)


def test_predict_scene_rejects_a_scene_smaller_than_a_tile():
    session = _ConstantSession(logit_value=1.0)
    with pytest.raises(ValueError, match="smaller than one"):
        predict_scene(session, np.zeros((5, 10, 10), dtype=np.float32), tile_size=40)


def test_predict_scene_rejects_wrong_dimensionality():
    session = _ConstantSession(logit_value=1.0)
    with pytest.raises(ValueError, match="channels, height, width"):
        predict_scene(session, np.zeros((100, 130), dtype=np.float32), tile_size=40)
