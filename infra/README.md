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

**`Dockerfile.api`**, **`Dockerfile.frontend`** (+ `nginx.frontend.conf`), and
**`docker-compose.yml`** — the rest of spec section 9's one-command stack.

- `Dockerfile.api` — multi-stage: `npm run build` in a build stage, then a second
  `npm ci --omit=dev` so the runtime image never ships `typescript`/`tsx`/`@types/*`
  (the compiled `dist/` output doesn't need them). Runs as `node:22-slim`'s built-in
  `node` user; healthcheck asserts on `/health`'s own `status` field, not just a 200 —
  same reasoning as `Dockerfile.serve`'s healthcheck, since a gateway that can't reach
  the inference service should read as unhealthy, not "fine."
- `Dockerfile.frontend` — builds the Vite app, then serves the static output from a
  plain `nginx:1.27-alpine` — no Node runtime in the final image. `VITE_API_BASE_URL` is
  a build arg (Vite env vars are compile-time, not runtime), pointed at the
  host-exposed gateway port rather than the in-compose service name, since it's read by
  the browser, which is outside the compose network. `nginx.frontend.conf` falls back
  unknown paths to `index.html` for future client-side routing.
- `docker-compose.yml` — `postgres` (the `postgis/postgis` image, precompiled, simpler
  than installing the extension into a plain Postgres image by hand — it applies
  `api/src/db/schema.sql` automatically via Postgres's own `docker-entrypoint-initdb.d/`
  convention), `redis`, `minio` (present per spec section 5's architecture diagram, but
  genuinely unused by any service's code yet — `inference` reads scenes from a local
  mounted directory, not S3, so this container runs with nothing talking to it; named
  honestly as reserved, not silently included for looks), `inference`, `api`, a separate
  `worker` service (same image, `dist/worker.js` entrypoint — an HTTP server and a
  long-running queue consumer scale independently), and `frontend`.
  **Honest gap this file cannot paper over**: there is no real trained checkpoint
  committed to the repo (correctly — it's a large build artifact that belongs on the
  cluster, not in git), so `inference` will fail its healthcheck and `/predict` will
  `503` until `MODELS_DIR`/`SCENES_DIR` are populated with either a real cluster export
  or the untrained local demo one (`inference/stage_demo_scene.py` +
  `inference/export_onnx.py`, see `inference/README.md`).

**Not yet build-tested.** Docker itself was unavailable in the sandboxed environment
this was written in — tried `colima` (a Docker-Desktop-free path to a real daemon)
specifically to get a real `docker compose up` run, but `colima start` failed on
`mkdir ~/.colima: operation not permitted`, a sandbox restriction on this session, not a
config mistake to work around. All four files are reviewed against the services' own
`.env.example`s and each other's port/env expectations, and `hadolint` was run against
`Dockerfile.api`/`Dockerfile.frontend` (same as the existing two) — a few info/warning-
level nits (non-numeric `USER`, shell-form `HEALTHCHECK CMD`) left unaddressed,
consistent with how `Dockerfile.serve`/`Dockerfile.train` already carry the same class
of nit. Treat all four Dockerfiles/compose file as reviewed-but-unverified until a real
build succeeds on a machine with Docker.