# inference/

The FastAPI service that runs a trained model on a real, full-size Sentinel-1 scene.

A raw Sentinel-1 scene is roughly 25,000 x 16,000 pixels, but the model only accepts small 512x512 tiles. So this service can't just call the model once — it has to:
- Split the scene into tiles.
- Run batched inference across those tiles (using the ONNX-exported model, not raw PyTorch, so this service stays lightweight).
- Stitch the per-tile predictions back into one full-scene flood mask.
- Convert that pixel mask into vector polygons, which is the format the database and map actually use.

This is the layer that turns "a trained model" into "a usable prediction on a real scene."
