"""
Tests for the Sen1Floods11 loader, run against tiny synthetic GeoTIFFs that
mimic the real dataset's file layout instead of the real ~4,800-chip
dataset (which isn't downloaded in this environment). Same idea as the
data contract tests: catch a bug here, on a laptop, in under a second,
rather than after real data is loaded.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from data.contract import LABEL_IGNORE, LABEL_NON_WATER, LABEL_WATER
from data.sen1floods11 import Sen1Floods11Dataset, read_split

CHIP_SIZE = 4  # real chips are 512x512; 4x4 is enough to exercise the code path

# An arbitrary but valid geotransform -- the tests don't care about real
# geographic coordinates, only that rasterio can write and read the file.
_TRANSFORM = from_origin(0, 0, 1, 1)


def _write_s1_image(path: Path, vv_db: float, vh_db: float) -> None:
    """Write a tiny 2-band GeoTIFF matching a real *_S1Hand.tif chip."""
    data = np.zeros((2, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    data[0] = vv_db
    data[1] = vh_db
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=2,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(data)


def _write_label(path: Path, values: np.ndarray) -> None:
    """Write a tiny 1-band GeoTIFF matching a real *_LabelHand.tif chip."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=1,
        dtype=np.int16,
        crs="EPSG:4326",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(values.astype(np.int16), 1)


def _make_fixture(tmp_path: Path) -> Path:
    """
    Build a miniature Sen1Floods11-shaped fixture: an image dir, a label
    dir, and a split CSV listing one chip, "Bolivia_1".
    """
    image_dir = tmp_path / "S1Hand"
    label_dir = tmp_path / "LabelHand"
    image_dir.mkdir()
    label_dir.mkdir()

    _write_s1_image(image_dir / "Bolivia_1_S1Hand.tif", vv_db=-10.0, vh_db=-20.0)

    label_values = np.full((CHIP_SIZE, CHIP_SIZE), LABEL_NON_WATER, dtype=np.int16)
    label_values[0, 0] = LABEL_WATER
    label_values[1, 1] = -1  # Sen1Floods11's no-data value
    _write_label(label_dir / "Bolivia_1_LabelHand.tif", label_values)

    split_csv = tmp_path / "flood_train_data.csv"
    split_csv.write_text("Bolivia_1_S1Hand.tif,Bolivia_1_LabelHand.tif\n")

    return tmp_path


def test_read_split_parses_pairs(tmp_path):
    split_csv = tmp_path / "split.csv"
    split_csv.write_text("a_S1Hand.tif,a_LabelHand.tif\nb_S1Hand.tif,b_LabelHand.tif\n")

    pairs = read_split(split_csv)

    assert pairs == [
        ("a_S1Hand.tif", "a_LabelHand.tif"),
        ("b_S1Hand.tif", "b_LabelHand.tif"),
    ]


def test_dataset_length_matches_split(tmp_path):
    root = _make_fixture(tmp_path)
    dataset = Sen1Floods11Dataset(
        image_dir=root / "S1Hand",
        label_dir=root / "LabelHand",
        split_csv=root / "flood_train_data.csv",
    )

    assert len(dataset) == 1


def test_dataset_loads_expected_image_channels(tmp_path):
    root = _make_fixture(tmp_path)
    dataset = Sen1Floods11Dataset(
        image_dir=root / "S1Hand",
        label_dir=root / "LabelHand",
        split_csv=root / "flood_train_data.csv",
    )

    sample = dataset[0]

    assert sample.chip_id == "Bolivia_1"
    assert sample.image.shape == (3, CHIP_SIZE, CHIP_SIZE)
    assert sample.image.dtype == np.float32
    assert np.allclose(sample.image[0], -10.0)  # VV_db
    assert np.allclose(sample.image[1], -20.0)  # VH_db
    assert np.allclose(sample.image[2], -10.0 / -20.0)  # VV_VH_ratio


def test_dataset_remaps_nodata_label_to_ignore_index(tmp_path):
    root = _make_fixture(tmp_path)
    dataset = Sen1Floods11Dataset(
        image_dir=root / "S1Hand",
        label_dir=root / "LabelHand",
        split_csv=root / "flood_train_data.csv",
    )

    sample = dataset[0]

    assert sample.label.dtype == np.int64
    assert sample.label[0, 0] == LABEL_WATER
    assert sample.label[1, 1] == LABEL_IGNORE  # was -1 in the source file
    assert sample.label[2, 2] == LABEL_NON_WATER
