"""
Entry point for fetching DEM chips for the weakly-labeled subset (see
data/download_sen1floods11_weak.py). Reuses fetch_dem_for_all_chips with
the weak-set naming (see its docstring for why dem_suffix stays
"_DEMHand.tif" even for weak chips).

Run directly: python -m data.fetch_dem_weak
"""

from pathlib import Path

from data.fetch_dem import fetch_dem_for_all_chips

if __name__ == "__main__":
    fetch_dem_for_all_chips(
        s1_dir=Path("datasets/sen1floods11_weak/S1Weak"),
        dest_dir=Path("datasets/sen1floods11_weak/DEMWeak"),
        s1_suffix="_S1Weak.tif",
    )
    print("done")
