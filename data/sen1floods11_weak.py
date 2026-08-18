"""
Loader for the Sen1Floods11 weakly-labeled subset (see
data/download_sen1floods11_weak.py). Mirrors data/sen1floods11.py's
HandLabeled loader (same file-per-layer GeoTIFF convention, same VV/VH ->
3-channel image, same -1 -> LABEL_IGNORE remap) but for the "Weak" file
suffix and the fact that there's no official train/val/test split for
this subset -- it exists purely for pretraining, so build_pairs below
just lists every chip and carves off a small held-out slice for
monitoring pretraining loss, not for any benchmark comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from data.contract import LABEL_IGNORE
from data.sen1floods11 import SEN1FLOODS11_LABEL_NODATA, _load_s1_image


@dataclass
class Sen1Floods11WeakSample:
    image: np.ndarray  # [3, H, W] float32: VV_db, VH_db, VV_VH_ratio
    label: np.ndarray  # [H, W] int64: 0 = not water, 1 = water, 255 = ignore
    chip_id: str


def _load_weak_label(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        label = src.read(1).astype(np.int64)
    label[label == SEN1FLOODS11_LABEL_NODATA] = LABEL_IGNORE
    return label


def list_weak_chip_ids(image_dir: str | Path) -> list[str]:
    """Every chip id with a downloaded S1Weak image, e.g. "Bolivia_1009032"."""
    image_dir = Path(image_dir)
    return sorted(p.name.split("_S1Weak")[0] for p in image_dir.glob("*_S1Weak.tif"))


def split_weak_chip_ids(
    chip_ids: list[str], val_fraction: float = 0.02, seed: int = 0
) -> tuple[list[str], list[str]]:
    """
    Carves a small held-out slice off the weakly-labeled set for
    monitoring pretraining loss/IoU trend -- not an official benchmark
    split (there isn't one for this subset), just enough signal to catch
    a pretraining run that's diverging before it wastes hours.
    """
    shuffled = list(chip_ids)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


class Sen1Floods11WeakDataset:
    """Iterates over a list of weakly-labeled chip ids."""

    def __init__(self, image_dir: str | Path, label_dir: str | Path, chip_ids: list[str]) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.chip_ids = chip_ids

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, index: int) -> Sen1Floods11WeakSample:
        chip_id = self.chip_ids[index]
        image = _load_s1_image(self.image_dir / f"{chip_id}_S1Weak.tif")
        label = _load_weak_label(self.label_dir / f"{chip_id}_S1OtsuLabelWeak.tif")
        return Sen1Floods11WeakSample(image=image, label=label, chip_id=chip_id)
