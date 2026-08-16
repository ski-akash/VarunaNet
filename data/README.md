# data/

Everything related to getting satellite data into a shape the models can train on.

This folder holds:
- Loaders for the Sen1Floods11 benchmark dataset (the labeled Sentinel-1 SAR chips we train and evaluate against).
- Fetchers for the real Sentinel-1 scenes over the Brahmaputra basin (pulled from Microsoft Planetary Computer's STAC API).
- Fetchers/processors for auxiliary layers: elevation-derived slope, HAND (Height Above Nearest Drainage), and the JRC permanent water layer. These exist to cut down false positives — a lot of things that aren't flood water still look dark in radar. `chip_terrain.py` ties slope+HAND together for one Sen1Floods11 chip, and caches them per chip id — shared by both the classical-baseline eval harness (`benchmarks/`) and real CNN training (`training/`), so the two never compute a chip's terrain in two different, silently-diverging ways.
- `compute_normalization_stats.py`, which computes and persists the real per-channel mean/std used everywhere else (training and, later, serving) — computed once on the training split only, never recomputed per-run.
- The shared "data contract": the exact tensor shape, channel order, and normalization convention every model in this project expects as input, so a change made here can't silently break a model downstream.

Nothing in here trains a model or makes a prediction — this layer's only job is turning raw sources into clean, normalized tensors.
