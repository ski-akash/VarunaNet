# infra/

Everything needed to make the whole system runnable with one command, on someone else's machine, without cluster access.

This folder holds:
- `Dockerfile.train` (CUDA base image, heavy — for training) and `Dockerfile.serve` (slim, no PyTorch, ONNX Runtime only — for serving) kept separate on purpose, since a 12GB serving image would defeat the point of a quick demo.
- `docker-compose.yml` that brings up the entire stack at once: frontend, API gateway, inference service, Redis, PostGIS, and MinIO, seeded with sample data and a pretrained checkpoint.
- CI config (GitHub Actions): lint, type-check, tests, and building both images on every push.
- A `Makefile` with the common commands (`make dev`, `make train-local`, `make bench`, `make eval-ai`) so nobody has to remember the exact long-form command for each task.

The goal stated in the spec is blunt but correct: reviewers won't build this repo by hand, so a one-command demo is what actually gets it looked at.


## Built so far

**`Dockerfile.serve`** and **`requirements-serve.txt`** — the slim serving image for
`inference/service.py`.

The split from `Dockerfile.train` is the whole point, and it is a measured one rather than
a stylistic preference:

| | Installed packages |
|---|---|
| Training-only (torch, torchvision, smp, transformers) | **647 MB** |
| Serving (onnxruntime, rasterio, shapely, pyproj, fastapi, numpy) | **211 MB** |

before any CUDA layers on the training side. The serving image needs none of the training
stack because the model arrives as ONNX and runs under ONNX Runtime.

That property is enforced, not just documented: `tests/test_serving_deps.py` imports the
service in a subprocess and fails if torch, torchvision, segmentation-models-pytorch or
transformers appear in `sys.modules`. It is exactly the kind of thing that regresses
silently — one convenient `from models.build_model import ...` in a serving module and the
image triples with nothing failing.

Other decisions in the image:

- **Only the modules the service imports are copied in**, not the whole repo. Copying
  everything would drag `training/` and `benchmarks/` in and make an accidental training
  import much easier to miss.
- **Model and scenes are mounted, not baked in.** A retrain should not require an image
  rebuild, and scenes are far too large to ship.
- **Runs as a non-root user**; the service only ever reads mounted scenes.
- **The healthcheck asserts on the payload, not just a 200.** `/health` deliberately
  returns `no_model` rather than failing, so a status-code-only check would keep a
  model-less container in the load balancer serving 503s.

**Not yet built.** Docker was unavailable in the environment this was written in, so it is
reviewed but unverified until a build actually succeeds. The dependency list was checked
against the service's real imported-module set.