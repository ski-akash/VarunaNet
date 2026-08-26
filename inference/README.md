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