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

import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from data.chip_terrain import TerrainCache, build_terrain_cache, get_terrain
from data.contract import validate_input_tensor, validate_label_tensor
from data.normalization import NormalizationStats, apply_normalization
from data.sen1floods11 import Sen1Floods11Dataset
from data.speckle import apply_speckle_noise


def _load_jrc_permanent_water(path: Path) -> np.ndarray:
    """
    Reads a *_JRCWaterHand.tif chip: 1 band, uint8, already binary (0 =
    not permanent water, 1 = permanent water -- confirmed directly by
    scanning the downloaded chips, not assumed). Left unnormalized: this
    channel is a baseline/change-detection signal for
    models/change_aware_unet.py, not a SAR measurement, so the
    normalization stats computed for the 5 SAR/terrain channels don't
    apply to it.
    """
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


class Sen1Floods11TorchDataset(Dataset):
    """
    A torch Dataset yielding data-contract-conformant (data/contract.py)
    5-channel input tensors, label tensors, and the chip id (e.g.
    "Bolivia_103757") for one Sen1Floods11 split. The chip id is what
    lets validation (training/train.py's run_validation) score results
    with benchmarks/metrics.py the same way the classical baselines were
    scored -- per-chip, with the flood event recoverable from the id.

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
        augment: bool = False,
        channel_indices: list[int] | None = None,
        jrc_dir: str | Path | None = None,
        speckle_prob: float = 0.0,
        speckle_looks: float = 4.0,
    ) -> None:
        self._dataset = Sen1Floods11Dataset(image_dir, label_dir, split_csv)
        self._dem_dir = Path(dem_dir)
        self._stats = normalization_stats
        self._terrain_cache = terrain_cache
        self._augment = augment
        # Speckle-noise augmentation (spec section 4.2), gated separately
        # from self._augment's flip logic so the two can be swept
        # independently -- speckle_prob=0.0 (default) is a strict no-op,
        # identical to this dataset's behavior before this knob existed.
        self._speckle_prob = speckle_prob
        self._speckle_looks = speckle_looks
        # When set, __getitem__ appends the JRC permanent-water mask as a
        # 6th channel (index 5), raw/unnormalized, for
        # models/change_aware_unet.py's baseline branch -- see
        # _load_jrc_permanent_water. None (default) keeps the standard
        # 5-channel contract every other model expects.
        self._jrc_dir = Path(jrc_dir) if jrc_dir is not None else None
        # Channel order is fixed: 0=VV_db, 1=VH_db, 2=VV_VH_ratio, 3=slope,
        # 4=HAND (see __getitem__'s concatenation order). None keeps all
        # 5 -- pass e.g. [0, 1] for a VV/VH-only ablation (spec section
        # 3.2's "does adding VV_VH_ratio/slope/HAND actually help, or is
        # it dead weight" question). Sliced *after* normalization, which
        # is computed per-channel, so dropping channels doesn't change
        # the normalization of the ones kept.
        self._channel_indices = channel_indices

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self._dataset[index]
        slope, hand = get_terrain(sample.chip_id, self._dem_dir, self._terrain_cache)

        full = np.concatenate([sample.image, slope[np.newaxis], hand[np.newaxis]], axis=0)

        # Applied here, before normalization, deliberately: the speckle
        # model operates on real dB backscatter values (data/speckle.py
        # converts to linear power internally), not on already
        # normalization-scaled ones, which have no physical meaning to
        # multiply against a noise model. Only VV_db (0) and VH_db (1)
        # are radar measurements -- slope/HAND are terrain, untouched.
        # VV_VH_ratio (2) is recomputed from the now-noisy VV/VH using
        # data/sen1floods11.py's own formula, so it stays internally
        # consistent with the channels it's derived from -- otherwise the
        # model would see a ratio channel computed from clean data
        # sitting next to VV/VH channels that no longer match it.
        if self._augment and self._speckle_prob > 0 and random.random() < self._speckle_prob:
            full[0] = apply_speckle_noise(full[0], self._speckle_looks)
            full[1] = apply_speckle_noise(full[1], self._speckle_looks)
            full[2] = full[0] / np.where(full[1] == 0, 1e-6, full[1])

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

        image_tensor = torch.from_numpy(full)
        label_tensor = torch.from_numpy(sample.label)

        # Sliced after validate_input_tensor, deliberately: a channel
        # ablation is a modeling choice, not a data-contract violation --
        # the full 5-channel tensor must still be valid before we decide
        # to drop any of it.
        if self._channel_indices is not None:
            image_tensor = image_tensor[self._channel_indices]

        if self._jrc_dir is not None:
            jrc = _load_jrc_permanent_water(self._jrc_dir / f"{sample.chip_id}_JRCWaterHand.tif")
            jrc_tensor = torch.from_numpy(jrc).unsqueeze(0)  # [1, H, W]
            image_tensor = torch.cat([image_tensor, jrc_tensor], dim=0)

        # Flips only (no rotation/crop): the terrain channels (slope,
        # HAND) and SAR channels all stay geometrically consistent under
        # a flip with no interpolation needed, unlike a rotation, which
        # would require resampling and could reintroduce NaN-adjacent
        # artifacts near the label's ignore borders.
        if self._augment:
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-1])
                label_tensor = torch.flip(label_tensor, dims=[-1])
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-2])
                label_tensor = torch.flip(label_tensor, dims=[-2])

        return image_tensor, label_tensor, sample.chip_id


def build_sen1floods11_dataset(
    data_root: str | Path,
    split_csv_name: str,
    normalization_stats: NormalizationStats,
    precompute_terrain: bool = True,
    augment: bool = False,
    channel_indices: list[int] | None = None,
    include_jrc_baseline: bool = False,
    speckle_prob: float = 0.0,
    speckle_looks: float = 4.0,
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
        augment=augment,
        channel_indices=channel_indices,
        jrc_dir=(data_root / "JRCWaterHand") if include_jrc_baseline else None,
        speckle_prob=speckle_prob,
        speckle_looks=speckle_looks,
    )
