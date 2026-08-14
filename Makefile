# Activate the venv first (.venv/Scripts/activate on Windows,
# .venv/bin/activate elsewhere) so `python` resolves to the project's own
# interpreter and dependencies.

.PHONY: bench

# Regenerates benchmarks/RESULTS.md: runs every baseline on the official
# split and under hold-one-event-out cross-validation, then writes the full
# models x metrics table. This is Phase 2's exit criterion -- the harness
# has to exist and run as one command before the first neural net is trained.
bench:
	python -m benchmarks.generate_results
