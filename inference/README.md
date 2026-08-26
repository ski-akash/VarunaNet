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
