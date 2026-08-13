# infra/

Everything needed to make the whole system runnable with one command, on someone else's machine, without cluster access.

This folder holds:
- `Dockerfile.train` (CUDA base image, heavy — for training) and `Dockerfile.serve` (slim, no PyTorch, ONNX Runtime only — for serving) kept separate on purpose, since a 12GB serving image would defeat the point of a quick demo.
- `docker-compose.yml` that brings up the entire stack at once: frontend, API gateway, inference service, Redis, PostGIS, and MinIO, seeded with sample data and a pretrained checkpoint.
- CI config (GitHub Actions): lint, type-check, tests, and building both images on every push.
- A `Makefile` with the common commands (`make dev`, `make train-local`, `make bench`, `make eval-ai`) so nobody has to remember the exact long-form command for each task.

The goal stated in the spec is blunt but correct: reviewers won't build this repo by hand, so a one-command demo is what actually gets it looked at.
