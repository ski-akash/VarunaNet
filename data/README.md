# data/

Everything related to getting satellite data into a shape the models can train on.

This folder holds:
- Loaders for the Sen1Floods11 benchmark dataset (the labeled Sentinel-1 SAR chips we train and evaluate against).
- Fetchers for real Sentinel-1 scenes over Assam, via Google Earth Engine (not Microsoft Planetary Computer, the spec's original preference — GEE was chosen once real acquisition planning showed it matches Sen1Floods11's own preprocessing chain exactly; see `fetch_assam_scene.py`'s module docstring and `VarunaNet_Spec.md` section 15.4).
- Fetchers/processors for auxiliary layers: elevation-derived slope, HAND (Height Above Nearest Drainage), and the JRC permanent water layer. These exist to cut down false positives — a lot of things that aren't flood water still look dark in radar. `chip_terrain.py` ties slope+HAND together for one Sen1Floods11 chip, and caches them per chip id — shared by both the classical-baseline eval harness (`benchmarks/`) and real CNN training (`training/`), so the two never compute a chip's terrain in two different, silently-diverging ways.
- `compute_normalization_stats.py`, which computes and persists the real per-channel mean/std used everywhere else (training and, later, serving) — computed once on the training split only, never recomputed per-run.
- The shared "data contract": the exact tensor shape, channel order, and normalization convention every model in this project expects as input, so a change made here can't silently break a model downstream.

Nothing in here trains a model or makes a prediction — this layer's only job is turning raw sources into clean, normalized tensors.

## Real Assam ingestion (Track B Step 3)

**`fetch_assam_scene.py`** — pulls a real, matched pre-flood/during-flood Sentinel-1
pair over an Assam AOI from Google Earth Engine (`COPERNICUS/S1_GRD`, default sigma0 --
**not** the RTC/gamma0 collection, which would introduce a silent domain shift against
Sen1Floods11, the training data; see the module's own docstring for why this is the
single highest-risk config detail in the whole pipeline). Both scenes share a relative
orbit and pass direction, so their backscatter geometry is directly comparable. Needs
`ee.Authenticate()` once (interactive) and a Google Cloud project with Earth Engine
enabled, passed via `--ee-project`/`EE_PROJECT_ID` -- never hardcoded.

Verified live, not assumed: a real pulled scene's VV/VH percentiles land almost exactly
inside this project's own Sen1Floods11-derived normalization stats (dry-season VV
median -10.2 dB vs. Sen1Floods11's -10.4 dB mean), strong direct evidence this is
genuinely the same sigma0 convention, not an accidental RTC pull.

**`build_assam_demo.py`** — the rest of the pipeline: fetches a matching Copernicus DEM
(slope + HAND) and the JRC Global Surface Water occurrence layer over the identical AOI
(so everything is already co-registered), runs this project's own validated classical
baseline (per-scene Otsu on VH + HAND/slope masking + permanent-water removal --
deliberately **not** the untrained local demo CNN checkpoint, which has random weights
and would report zero flood on real data), vectorizes the result, and aggregates it to
real Assam districts. Writes `frontend/public/data/assam_flood_demo.json` (district
summary) and `.geojson` (flood polygons), served the same way
`assam_districts.geojson` already is.

**Real result** (Majuli-area AOI, 2020 monsoon flood vs. a January 2020 dry-season
pass, same orbit): raw VH backscatter alone shows water-like coverage jumping from 25%
to 69% between the two passes; the full Otsu+HAND+permanent-water pipeline reports
13,465 ha of flood extent, real intersections against real district boundaries putting
Lakhimpur (3.22% of its area) and Jorhat (1.40%) as affected -- both real districts,
not "Majuli" (which isn't its own polygon in this project's 2011-census boundary data;
Majuli became a district in 2016, after that census, so the AOI's true geometry falls
inside its old parent districts' boundaries, exactly as the spatial join should handle).

Both scripts are honestly scoped to one small AOI (~16km), not full-state coverage --
see `frontend/src/lib/assamFloodDemo.ts` for how the frontend represents that limit
rather than implying more coverage than exists.
