"""
Torch Dataset wrapping the weakly-labeled Sen1Floods11 subset (see
data/sen1floods11_weak.py) for pretraining -- same 5-channel data
contract (VV_db, VH_db, VV_VH_ratio, slope, HAND), same normalization
stats, same NaN handling, and same flip augmentation as
training/sen1floods11_dataset.py's hand-labeled dataset, so a model
pretrained here can be fine-tuned on hand-labeled data with no
architecture or preprocessing mismatch.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.chip_terrain import TerrainCache, build_terrain_cache, get_terrain
from data.contract import validate_input_tensor, validate_label_tensor
from data.normalization import NormalizationStats, apply_normalization
from data.sen1floods11_weak import Sen1Floods11WeakDataset, list_weak_chip_ids, split_weak_chip_ids


class Sen1Floods11WeakTorchDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        dem_dir: str | Path,
        chip_ids: list[str],
        normalization_stats: NormalizationStats,
        terrain_cache: TerrainCache | None = None,
        augment: bool = False,
    ) -> None:
        self._dataset = Sen1Floods11WeakDataset(image_dir, label_dir, chip_ids)
        self._dem_dir = Path(dem_dir)
        self._stats = normalization_stats
        self._terrain_cache = terrain_cache
        self._augment = augment

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self._dataset[index]
        slope, hand = get_terrain(sample.chip_id, self._dem_dir, self._terrain_cache)

        full = np.concatenate([sample.image, slope[np.newaxis], hand[np.newaxis]], axis=0)
        full = apply_normalization(full, self._stats)
        full = np.nan_to_num(full, nan=0.0)

        validate_input_tensor(full)
        validate_label_tensor(sample.label)

        image_tensor = torch.from_numpy(full)
        label_tensor = torch.from_numpy(sample.label)

        if self._augment:
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-1])
                label_tensor = torch.flip(label_tensor, dims=[-1])
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-2])
                label_tensor = torch.flip(label_tensor, dims=[-2])

        return image_tensor, label_tensor, sample.chip_id


def build_sen1floods11_weak_dataset(
    data_root: str | Path,
    normalization_stats: NormalizationStats,
    split: str = "train",
    val_fraction: float = 0.02,
    split_seed: int = 0,
    precompute_terrain: bool = True,
    augment: bool = False,
) -> Sen1Floods11WeakTorchDataset:
    """
    split: "train" or "val" -- see data/sen1floods11_weak.py's
    split_weak_chip_ids for what this held-out slice is (and isn't) for.
    """
    data_root = Path(data_root)
    image_dir = data_root / "S1Weak"
    label_dir = data_root / "S1OtsuLabelWeak"
    dem_dir = data_root / "DEMWeak"

    all_chip_ids = list_weak_chip_ids(image_dir)
    train_ids, val_ids = split_weak_chip_ids(all_chip_ids, val_fraction=val_fraction, seed=split_seed)
    chip_ids = train_ids if split == "train" else val_ids

    terrain_cache = None
    if precompute_terrain:
        terrain_cache = build_terrain_cache(chip_ids, dem_dir)

    return Sen1Floods11WeakTorchDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        dem_dir=dem_dir,
        chip_ids=chip_ids,
        normalization_stats=normalization_stats,
        terrain_cache=terrain_cache,
        augment=augment,
    )
