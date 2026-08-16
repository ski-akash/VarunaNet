"""
The real Sen1Floods11 dataset (spec section 3.2's Track A), wired into the
training loop as the "sen1floods11" dataset.name option -- the option
training/train.py's build_dataloader has raised NotImplementedError for
since the training loop was first built, specifically so the loop itself
could be written and tested against tiny synthetic tensors before ever
touching real chips (spec section 8).

Combines three things data/sen1floods11.py deliberately keeps apart (see
its own module docstring):
  1. the raw 3-channel S1 image + label (VV_db, VH_db, VV_VH_ratio) from
     data/sen1floods11.py,
  2. slope + HAND derived from each chip's own DEM (data/chip_terrain.py),
     the auxiliary layers that turn "water detection" into "flood
     detection" (spec section 3.2),
  3. per-channel normalization (data/normalization.py), using stats
     computed once on the training split and persisted to disk (see
     data/compute_normalization_stats.py) -- never recomputed per-run,
     which is exactly the train/serve mismatch spec section 3.3 calls
     "the most common way this class of project fails".

Terrain is precomputed once per chip up front (data/chip_terrain.py's
build_terrain_cache), not inside __getitem__: real training reads the
same ~250 training chips again every single epoch, and pysheds' HAND flow
routing is expensive enough that recomputing it on the fly would make
terrain computation the dominant cost of training instead of the model
itself -- the exact problem benchmarks/hold_one_event_out.py already
solved the same way, for the same reason (most chips get read many times
over across its 11 folds).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.chip_terrain import TerrainCache, build_terrain_cache, get_terrain
from data.contract import validate_input_tensor, validate_label_tensor
from data.normalization import NormalizationStats, apply_normalization
from data.sen1floods11 import Sen1Floods11Dataset


class Sen1Floods11TorchDataset(Dataset):
    """
    A torch Dataset yielding data-contract-conformant (data/contract.py)
    5-channel input tensors and label tensors for one Sen1Floods11 split.

    terrain_cache is optional but strongly recommended for anything beyond
    a single pass over the data -- see this module's docstring. Pass one
    built with data.chip_terrain.build_terrain_cache(chip_ids, dem_dir)
    over every chip id this dataset will actually read.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        dem_dir: str | Path,
        split_csv: str | Path,
        normalization_stats: NormalizationStats,
        terrain_cache: TerrainCache | None = None,
    ) -> None:
        self._dataset = Sen1Floods11Dataset(image_dir, label_dir, split_csv)
        self._dem_dir = Path(dem_dir)
        self._stats = normalization_stats
        self._terrain_cache = terrain_cache

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._dataset[index]
        slope, hand = get_terrain(sample.chip_id, self._dem_dir, self._terrain_cache)

        full = np.concatenate([sample.image, slope[np.newaxis], hand[np.newaxis]], axis=0)
        full = apply_normalization(full, self._stats)

        # Real chips carry NaN by design, not as a rare edge case -- every
        # single chip's HAND channel has NaN border pixels from
        # flow-routing (data/hand.py), and some chips have NaN VV/VH at
        # scene edges too. A NaN anywhere in a convolution's receptive
        # field poisons that whole output region and cascades through
        # every later layer, so it has to be resolved to a real number
        # here, before the tensor ever reaches the model. 0.0 lands
        # exactly at "this channel's mean" post-normalization -- treating
        # an unknown pixel as "typical for this channel" rather than
        # leaving it undefined, the same spirit as this project's other
        # NaN-pixel conventions (benchmarks/random_forest.py defaults
        # individual NaN pixels to "not water" rather than crashing).
        full = np.nan_to_num(full, nan=0.0)

        # Fails loudly on a shape/dtype mismatch (or, now, a NaN that
        # somehow survived the line above) here rather than letting it
        # reach the model as a confusing error several layers deep --
        # same reasoning as every other contract check in this project.
        validate_input_tensor(full)
        validate_label_tensor(sample.label)

        return torch.from_numpy(full), torch.from_numpy(sample.label)


def build_sen1floods11_dataset(
    data_root: str | Path,
    split_csv_name: str,
    normalization_stats: NormalizationStats,
    precompute_terrain: bool = True,
) -> Sen1Floods11TorchDataset:
    """
    Convenience constructor matching Sen1Floods11's on-disk layout (see
    VarunaNet_Spec.md's Environment notes): data_root/S1Hand,
    data_root/LabelHand, data_root/DEMHand, data_root/splits/<split_csv_name>.

    precompute_terrain=False is for tests that only want to exercise a
    couple of chips and don't want to pay for a terrain cache covering the
    whole split up front.
    """
    data_root = Path(data_root)
    dem_dir = data_root / "DEMHand"
    split_csv = data_root / "splits" / split_csv_name

    terrain_cache = None
    if precompute_terrain:
        from data.sen1floods11 import read_split

        chip_ids = [
            image_filename.split("_S1Hand")[0] for image_filename, _ in read_split(split_csv)
        ]
        terrain_cache = build_terrain_cache(chip_ids, dem_dir)

    return Sen1Floods11TorchDataset(
        image_dir=data_root / "S1Hand",
        label_dir=data_root / "LabelHand",
        dem_dir=dem_dir,
        split_csv=split_csv,
        normalization_stats=normalization_stats,
        terrain_cache=terrain_cache,
    )
