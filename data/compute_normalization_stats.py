"""
Computes per-channel normalization statistics (data/normalization.py) over
the real Sen1Floods11 training split, and persists them to disk.

Run once (python -m data.compute_normalization_stats) after the dataset
and its DEMs are downloaded locally (see VarunaNet_Spec.md's Environment
notes: `python -m data.download_sen1floods11`, then `python -m
data.fetch_dem`). The output file is committed to the repo -- small (a
mean and std per channel, five numbers each), and needed by both training
and any future serving code, which must never recompute its own stats
(spec section 3.3: "A silent train/serve normalization mismatch is the
most common way this class of project fails").

Only ever run against the *training* split -- see
compute_normalization_stats's own docstring for why touching val/test data
here would leak information about those splits into the model.
"""

from pathlib import Path

import numpy as np

from data.chip_terrain import build_terrain_cache
from data.normalization import NormalizationStats, compute_normalization_stats
from data.sen1floods11 import Sen1Floods11Dataset, read_split

DEFAULT_OUTPUT_PATH = Path("data/sen1floods11_normalization_stats.json")


def compute_and_save_training_stats(
    data_root: str | Path = "datasets/sen1floods11",
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> NormalizationStats:
    data_root = Path(data_root)
    split_csv = data_root / "splits" / "flood_train_data.csv"

    dataset = Sen1Floods11Dataset(
        image_dir=data_root / "S1Hand",
        label_dir=data_root / "LabelHand",
        split_csv=split_csv,
    )

    chip_ids = [image_filename.split("_S1Hand")[0] for image_filename, _ in read_split(split_csv)]
    terrain_cache = build_terrain_cache(chip_ids, data_root / "DEMHand")

    tensors = []
    for i in range(len(dataset)):
        sample = dataset[i]
        slope, hand = terrain_cache[sample.chip_id]
        full = np.concatenate([sample.image, slope[np.newaxis], hand[np.newaxis]], axis=0)
        tensors.append(full)
        print(f"[stats {i + 1}/{len(dataset)}] {sample.chip_id}")

    stats = compute_normalization_stats(tensors)
    stats.save(output_path)
    print(f"saved normalization stats to {output_path}")
    return stats


if __name__ == "__main__":
    compute_and_save_training_stats()
