"""
Downloads the Sen1Floods11 weakly-labeled subset (4,384 chips) from the
dataset's public Google Cloud Storage bucket -- the larger set
data/download_sen1floods11.py deliberately skips (see its own docstring).

Used for pretraining before fine-tuning on the small hand-labeled set:
the weak labels are Otsu-threshold water masks (S1OtsuLabelWeak), not
human-verified, so they're noisier than LabelHand, but there are ~10x
more of them -- literature on this dataset (Bonafilia et al. 2020)
consistently finds pretraining on weak labels then fine-tuning on hand
labels beats training on hand labels alone.

Only S1Weak (the SAR image) and S1OtsuLabelWeak (its Otsu-derived weak
label) are fetched, matching this project's SAR-only scope -- S2Weak and
S2IndexLabelWeak (optical-derived) are skipped for the same reason
download_sen1floods11.py skips Sentinel-2 entirely.

Run directly (`python -m data.download_sen1floods11_weak`) to fetch
everything into `datasets/sen1floods11_weak/`. Safe to re-run: already-
downloaded files are skipped.
"""

import concurrent.futures
import json
import urllib.request
from pathlib import Path

BUCKET_BASE = "https://storage.googleapis.com/sen1floods11"
LISTING_API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"

WEAK_FOLDERS = ["S1Weak", "S1OtsuLabelWeak"]


def _list_objects(prefix: str) -> list[str]:
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
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BUCKET_BASE}/{object_name}", dest_path)


def download_weakly_labeled(dest_root: Path, max_workers: int = 16) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)

    for folder in WEAK_FOLDERS:
        prefix = f"v1.1/data/flood_events/WeaklyLabeled/{folder}/"
        object_names = _list_objects(prefix)
        print(f"{folder}: {len(object_names)} files")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_download_one, name, dest_root / folder, prefix)
                for name in object_names
            ]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                future.result()
                if (i + 1) % 500 == 0:
                    print(f"  {folder}: {i + 1}/{len(object_names)}")


if __name__ == "__main__":
    download_weakly_labeled(Path("datasets/sen1floods11_weak"))
    print("done")
