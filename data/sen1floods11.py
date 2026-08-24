"""
Loader for the Sen1Floods11 benchmark dataset (Track A of the data
strategy).

Sen1Floods11 ships as one GeoTIFF file per chip per "layer" (the
Sentinel-1 image, the hand-labeled ground truth, etc), all sharing a
filename prefix, e.g.:

    Bolivia_103757_S1Hand.tif      <- Sentinel-1 image, 2 bands: VV, VH (dB)
    Bolivia_103757_LabelHand.tif   <- hand-labeled ground truth, 1 band:
                                       -1 = no data, 0 = not water, 1 = water

Which chips belong to train/val/test is decided by the dataset's official
split lists, not by us -- re-splitting randomly would make results
incomparable to published Sen1Floods11 benchmarks, which is the entire
point of using this dataset for credibility.

Note on channels: the data contract (see data/contract.py) wants 5
channels per input tensor (VV_db, VH_db, VV_VH_ratio, slope, HAND), but
Sen1Floods11's own files only carry VV and VH. slope and HAND come from a
separate DEM/HAND co-registration step (a later task) that gets
concatenated onto this loader's output before training. So on its own,
this loader intentionally returns a partial 3-channel tensor -- it is not
yet a contract-valid input tensor by itself, and isn't meant to be checked
against validate_input_tensor until the auxiliary layers are joined in.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from data.contract import LABEL_IGNORE

# Sen1Floods11 marks no-data label pixels as -1; our contract uses 255 for
# the same idea (see data/contract.py). Naming this instead of leaving a
# bare -1 in the code makes the remap below easy to spot.
SEN1FLOODS11_LABEL_NODATA = -1


@dataclass
class Sen1Floods11Sample:
    image: np.ndarray  # [3, H, W] float32: VV_db, VH_db, VV_VH_ratio
    label: np.ndarray  # [H, W] int64: 0 = not water, 1 = water, 255 = ignore
    chip_id: str


def read_split(split_csv: str | Path) -> list[tuple[str, str]]:
    """
    Read an official Sen1Floods11 split file: a CSV with no header, where
    each row is `image_filename,label_filename`.
    """
    pairs = []
    with Path(split_csv).open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            image_filename, label_filename = row
            pairs.append((image_filename.strip(), label_filename.strip()))
    return pairs


def _load_s1_image(path: Path) -> np.ndarray:
    """Read a Sen1Floods11 S1 GeoTIFF (VV, VH bands in dB) and derive the ratio channel."""
    with rasterio.open(path) as src:
        bands = src.read().astype(np.float32)  # [2, H, W]: VV_db, VH_db

    vv_db, vh_db = bands[0], bands[1]
    # VV_db and VH_db are already logarithmic (decibel) quantities, so the
    # power ratio VV/VH is expressed in dB by SUBTRACTING them, not dividing
    # them -- division in linear power space is subtraction in log space, the
    # same reason two magnitudes in dB combine by adding/subtracting rather
    # than multiplying/dividing. The original `vv_db / vh_db` had no physical
    # meaning as a ratio at all, and directly dividing two values that
    # legitimately land close to (but never exactly) zero blew this channel
    # up to extreme outliers -- measured on real test chips at -1914 to
    # +5289 after normalization, versus every other channel's roughly
    # +/-10 to +/-33 range. That numerical garbage is almost certainly why
    # this channel measured ~0 correlation with the water label (0.0096,
    # versus VV_db/VH_db's ~-0.55/-0.61) and why every ablation that dropped
    # it looked harmless -- the model had nothing usable to learn from it in
    # the first place, so removing it cost nothing. Subtraction has no
    # zero-denominator failure mode, so the previous guard is gone too.
    vv_vh_ratio = vv_db - vh_db

    return np.stack([vv_db, vh_db, vv_vh_ratio], axis=0)


def _load_label(path: Path) -> np.ndarray:
    """Read a Sen1Floods11 hand-labeled GeoTIFF and remap its no-data value to our contract's."""
    with rasterio.open(path) as src:
        label = src.read(1).astype(np.int64)  # [H, W]

    label[label == SEN1FLOODS11_LABEL_NODATA] = LABEL_IGNORE
    return label


class Sen1Floods11Dataset:
    """
    Iterates over one Sen1Floods11 split (train, val, or test).

    `image_dir` and `label_dir` point at the folders holding the S1Hand
    and LabelHand GeoTIFFs; `split_csv` is one of the dataset's official
    split files, so the same three train/val/test lists Sen1Floods11
    ships with are what decide membership here.
    """

    def __init__(self, image_dir: str | Path, label_dir: str | Path, split_csv: str | Path) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.pairs = read_split(split_csv)

    @classmethod
    def from_pairs(
        cls, image_dir: str | Path, label_dir: str | Path, pairs: list[tuple[str, str]]
    ) -> "Sen1Floods11Dataset":
        """
        Build a dataset from an explicit list of (image_filename,
        label_filename) pairs instead of one official split file.

        Needed for evaluation protocols that repartition chips by
        something other than the official train/val/test assignment --
        e.g. hold-one-event-out cross-validation (see
        benchmarks/hold_one_event_out.py), which pools chips across all
        three official splits and then regroups them by flood event.
        """
        dataset = cls.__new__(cls)
        dataset.image_dir = Path(image_dir)
        dataset.label_dir = Path(label_dir)
        dataset.pairs = pairs
        return dataset

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Sen1Floods11Sample:
        image_filename, label_filename = self.pairs[index]

        image = _load_s1_image(self.image_dir / image_filename)
        label = _load_label(self.label_dir / label_filename)

        # e.g. "Bolivia_103757_S1Hand.tif" -> "Bolivia_103757"
        chip_id = image_filename.split("_S1Hand")[0]

        return Sen1Floods11Sample(image=image, label=label, chip_id=chip_id)
