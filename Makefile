# Activate the venv first (.venv/Scripts/activate on Windows,
# .venv/bin/activate elsewhere) so `python` resolves to the project's own
# interpreter and dependencies.

.PHONY: bench dev

# Regenerates benchmarks/RESULTS.md: runs every baseline on the official
# split and under hold-one-event-out cross-validation, then writes the full
# models x metrics table. This is Phase 2's exit criterion -- the harness
# has to exist and run as one command before the first neural net is trained.
bench:
	python -m benchmarks.generate_results

# Brings up the whole stack (frontend, gateway, worker, inference, Redis,
# Postgres+PostGIS, MinIO) with no cluster access -- spec section 9. See
# infra/README.md for the real, current gap: MODELS_DIR/SCENES_DIR need a
# real or demo model staged before `inference` (and anything downstream of
# it) is actually healthy.
dev:
	docker compose -f infra/docker-compose.yml up --build
