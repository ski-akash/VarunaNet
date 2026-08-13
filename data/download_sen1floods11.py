"""
Downloads the Sen1Floods11 hand-labeled subset from the dataset's public
Google Cloud Storage bucket.

Only the hand-labeled subset (446 chips) is pulled here -- it's what
Phase 2's classical baselines and benchmark harness need. Two things are
deliberately skipped to keep this a quick, disk-friendly download:
- The much larger weakly-labeled set, which exists for full-scale CNN
  training in later phases, not for baselines.
- The Sentinel-2 imagery, since this project is SAR-only.

Run directly (`python -m data.download_sen1floods11`) to fetch everything
into `datasets/sen1floods11/` at the repo root. Safe to re-run: already-
downloaded files are skipped, so an interrupted download can just be
started again.
"""

import concurrent.futures
import json
import urllib.request
from pathlib import Path

BUCKET_BASE = "https://storage.googleapis.com/sen1floods11"
LISTING_API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"

# The four HandLabeled subfolders we actually use:
#   S1Hand          - Sentinel-1 VV/VH chips
#   LabelHand       - hand-labeled ground truth
#   JRCWaterHand    - JRC permanent water mask, pre-cropped to each chip
#   S1OtsuLabelHand - precomputed Otsu-threshold labels, useful as a
#                     second opinion when building the Otsu baseline
HAND_LABELED_FOLDERS = ["S1Hand", "LabelHand", "JRCWaterHand", "S1OtsuLabelHand"]

SPLIT_FILES = ["flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv"]


def _list_objects(prefix: str) -> list[str]:
    """List every object name under a bucket prefix, following pagination."""
    names = []
    page_token = None
    while True:
        url = f"{LISTING_API}?prefix={prefix}"
        if page_token:
            url += f"&pageToken={page_token}"
        with urllib.request.urlopen(url) as response:
            payload = json.load(response)
        names.extend(item["name"] for item in payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return names


def _download_one(object_name: str, dest_root: Path, prefix: str) -> None:
    relative_path = object_name[len(prefix) :]
    dest_path = dest_root / relative_path
    if dest_path.exists():
        return  # already downloaded -- makes the whole script resumable
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BUCKET_BASE}/{object_name}", dest_path)


def download_hand_labeled(dest_root: Path, max_workers: int = 16) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)

    for folder in HAND_LABELED_FOLDERS:
        prefix = f"v1.1/data/flood_events/HandLabeled/{folder}/"
        object_names = _list_objects(prefix)
        print(f"{folder}: {len(object_names)} files")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_download_one, name, dest_root / folder, prefix)
                for name in object_names
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # re-raise if any single download failed

    splits_dir = dest_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    for filename in SPLIT_FILES:
        url = f"{BUCKET_BASE}/v1.1/splits/flood_handlabeled/{filename}"
        urllib.request.urlretrieve(url, splits_dir / filename)
    print(f"splits: {len(SPLIT_FILES)} files")


if __name__ == "__main__":
    download_hand_labeled(Path("datasets/sen1floods11"))
    print("done")
