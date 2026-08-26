# inference/

The FastAPI service that runs a trained model on a real, full-size Sentinel-1 scene.

A raw Sentinel-1 scene is roughly 25,000 x 16,000 pixels, but the model only accepts small 512x512 tiles. So this service can't just call the model once — it has to:
- Split the scene into tiles.
- Run batched inference across those tiles (using the ONNX-exported model, not raw PyTorch, so this service stays lightweight).
- Stitch the per-tile predictions back into one full-scene flood mask.
- Convert that pixel mask into vector polygons, which is the format the database and map actually use.

This is the layer that turns "a trained model" into "a usable prediction on a real scene."

## Built so far

**`export_onnx.py`** — converts a trained checkpoint into a single self-contained ONNX
file, and verifies it. Three things it does that a bare `torch.onnx.export` call does not:

- **Checks the graph actually computes the same function.** Export succeeding only proves
  a file was written. The exported model is run against PyTorch on a fresh random input
  and must agree to within 1e-4 on logits, or the file is deleted rather than left on disk
  looking usable. On the real U-Net++ checkpoint the largest disagreement is 7.6e-06.
- **Keeps the batch axis dynamic, and proves it.** Verification uses a different batch
  size than the one traced, so a batch dimension silently frozen at export time fails
  here instead of in the tiler.
- **Folds weights back into the file.** torch's exporter writes them to a sidecar
  `<name>.onnx.data`, leaving a ~377KB `.onnx` that looks like a complete model, copies
  like one, and loads with no weights. That split is inlined and the sidecar removed.

Measured on U-Net++ (job 2218): a 313MB checkpoint becomes a 105MB ONNX file — the
checkpoint also carries optimizer and scheduler state that serving has no use for.


**`tiling.py`** — turns the model into a scene-level prediction. Splits a scene into
overlapping 512x512 tiles, batches them through an ONNX session, and stitches the
per-tile logits back into one scene-sized mask.

The interesting parts are all at the joins:

- **Tiles overlap** (128px by default). A conv model has far less context at a tile's
  edge than its centre -- the receptive field runs off the end into padding rather than
  neighbouring water -- so predictions are systematically worse near borders. Butt-jointed
  tiles put those weak edges against each other and produce a visible grid of seams.
- **A tapered blend, not a plain average.** Each tile's contribution is weighted by a map
  peaking at its centre, so wherever tiles overlap the one that can see the most context
  dominates. Weights are accumulated alongside the logits and divided out, giving a
  weighted mean.
- **Blending happens on logits, then thresholds once.** Averaging binary masks throws away
  confidence and leaves ragged half-committed boundaries -- the same reasoning as
  logit-averaged ensembling in `training/evaluate_ensemble.py`.
- **The last tile is pulled flush with the scene edge** rather than padding or leaving a
  strip unpredicted. Scene dimensions are essentially never a multiple of 512, so this is
  the normal case.

Measured end to end with the real U-Net++ export on this project's dev machine (CPU,
ONNX Runtime): **~1.0 s per 512x512 tile**, which extrapolates to roughly **an hour for a
full 25,000 x 16,000 scene**. That is the number the FP16/INT8 quantization work in spec
section 5 has to improve on, and it is why the pipeline is queued (BullMQ) rather than
synchronous.

**`quantize.py`** — dynamic INT8 quantization, measured rather than asserted.

Measured on the real U-Net++ export against **16 real Sen1Floods11 test chips**
(CPU, ONNX Runtime):

| | Size | Latency | Mask pixels changed |
|---|---|---|---|
| FP32 | 104.7 MB | 1208.6 ms/tile | — |
| INT8 | 26.4 MB | 708.1 ms/tile | 0.258% |

**3.96x smaller, 1.71x faster**, and pooled IoU over those chips moves 0.5540 -> 0.5563 —
within noise, and if anything slightly up. For a serving path that is a clear win: a full
25,000 x 16,000 scene drops from roughly an hour to about 35 minutes on CPU.

Two deliberate choices:

- **The accuracy metric is disagreement on the thresholded mask**, not MSE on logits.
  What ships is a water/not-water decision; a logit moving 4.1 -> 4.0 changes nothing
  while one crossing zero flips a pixel, and MSE averages those together.
- **Benchmarked on real chips, never random noise.** Random input reported *2.80x faster
  and 0.135% pixels changed* for the same pair of models — overstating the speedup and
  understating the damage simultaneously. Dynamic quantization keys off activation
  ranges, and random noise has none of SAR's actual distribution.

Dynamic rather than static quantization: static is more accurate but needs a calibration
dataset, making it a data-dependent build step. Whether dynamic's accuracy cost is
acceptable is what the table above answers.

**`streaming.py`** — full-scene inference without holding the scene.

`tiling.predict_scene` takes an in-memory array and accumulates scene-sized float64
buffers. For a real 25,000 x 16,000 scene that is:

| | In-memory | Streaming |
|---|---|---|
| Input | 8.00 GB | one 512x512 tile at a time |
| Stitch accumulators | 6.40 GB | **131 MB** |
| Peak | **~14.4 GB** | bounded, independent of scene height |

14.4 GB is more than the serving container will have, so the in-memory path is fine for
a subset and unusable for the thing the service exists to do.

The observation that makes it work: tiles are laid out in rows, so once the next row of
tiles starts below row R, no future tile can touch anything above R. Those rows are final
— threshold them, hand them to the writer, drop them. Only a band of `tile_size` rows is
ever live, and the accumulators stop depending on scene height entirely.

The reader and writer are callables, not rasterio handles, so the streaming logic is
testable without raster fixtures and a caller can back it with a local GeoTIFF, a COG, or
object storage without this module knowing.

**Verified to produce byte-identical output to the in-memory path** across scenes that
are exact tile multiples, scenes that are not, a single-tile scene, and a very wide one.
That equivalence is the test that matters: the design rests on "rows above the next tile
row can never change again", and an off-by-one there produces a horizontal seam every
tile row — plausible enough at a glance to ship unnoticed.

(The measured saving on a small test scene is only ~1.5x, because the test harness's
`collect_bands` assembles the whole mask in memory, which is exactly what streaming
exists to avoid. The saving scales with scene height; the table above is the real case.)

**`vectorize.py`** — pixel mask to georeferenced polygons, which is what the database and
map actually consume. The dashboard never queries pixels: answering "how much of Nagaon is
flooded" against a raster means rasterizing district boundaries and counting per query,
whereas against polygons in PostGIS it is one spatial join.

Three steps, and the last two are why this is not a one-line call to
`rasterio.features.shapes`:

- **Drop specks.** A real SAR mask contains thousands of isolated one- and two-pixel
  blobs. Filtered by real area (default 1 ha), not pixel count, so the threshold means the
  same thing at any resolution.
- **Simplify.** Traced polygons follow pixel edges, so every boundary is a staircase with
  a vertex every 10m. Tolerance is half a pixel, with `preserve_topology` so simplification
  cannot produce the self-intersecting rings PostGIS rejects on insert.
- **Areas are geodesic on a geographic CRS.** Sen1Floods11 chips are EPSG:4326, so the
  affine is in *degrees*. Taking its determinant as an area gives ~8e-09 per pixel, every
  polygon falls under any sane threshold, and the scene silently reports zero flooding.
  That is not hypothetical — the first real-chip run produced **0.0 hectares from a scene
  77.6% covered in predicted water**, all 14 regions discarded as specks. The CRS is now
  inspected and geographic scenes get true ellipsoidal areas via pyproj.

Verified end to end on `Mekong_1443339` (73.5% water by label) through the INT8 model:
**19.75 km² across 3 polygons**, 11 specks dropped, against ~20.34 km² implied by the raw
pixel count — the ~3% difference being exactly the specks and simplification.

Holes are preserved: a dry patch inside a flooded area is real information, and dropping
interiors would systematically overstate flood extent.

**`districts.py`** — aggregates flood polygons to districts, which is the form every
question is actually asked in. Nobody asks about geometry; they ask "which districts are
worst hit this week".

Reuses `frontend/public/geo/assam_districts.geojson`, the same file the map draws, so the
dashboard's boundaries and the reported statistics can never disagree about where a
district is.

**Ranked by share of the district, not absolute area.** Assam's districts differ in size
by more than 10x, so ranking by flooded km² mostly reproduces a ranking of district size.
On a synthetic Brahmaputra-valley flood band:

| District | Flooded | % of district |
|---|---|---|
| Kamrup Metropolitan | 87,601 ha | **72.6%** |
| Marigaon | 111,689 ha | 72.2% |
| Barpeta | 127,086 ha | 55.0% |
| Nagaon | 144,372 ha | 36.2% |

Nagaon has the most flooded hectares of any district and is only sixth worst affected. A
small district most of the way under water is the emergency.

Other deliberate choices:

- **Real intersections, not centroid or bbox tests.** Floods follow river courses and
  routinely straddle several districts; anything coarser assigns a whole flood to one
  district and reports zero for its neighbours.
- **Flood polygons are unioned first**, so overlapping polygons cannot double-count the
  same ground.
- **Untouched districts are reported as zero, not omitted.** "Dhemaji is not flooded" is a
  real answer, and a list that drops dry districts cannot distinguish them from districts
  the scene never covered.
- **Geodesic areas**, for the same reason as `vectorize.py`. Validated against reality:
  the 27 districts sum to **78,470 km²** against Assam's actual **78,438 km²**, and
  Karbi Anglong comes out as the largest district, which it is.

`summarize()` returns a compact JSON payload — the shape an API response or a grounded
LLM tool call would return, where every figure traces to a computed value rather than a
model's recollection (spec section 6.1).