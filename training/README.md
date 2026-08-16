# training/

The actual training loop, plus everything needed to run it on a remote SLURM GPU cluster instead of a laptop.

This folder holds:
- The config-driven training loop (Hydra/YAML configs — no hardcoded paths or hyperparameters, so the same code runs locally on a tiny synthetic dataset or on the cluster on the full real one, switched with a single `dataset=synthetic`/`dataset=sen1floods11` override).
- `sen1floods11_dataset.py`, the real dataset: wraps `data/sen1floods11.py`'s raw loader with terrain (slope+HAND) and normalization to produce data-contract-conformant tensors, with a precomputed terrain cache so a multi-epoch run doesn't recompute HAND flow-routing for the same chip every epoch.
- Checkpointing and automatic resume logic, since cluster jobs can get preempted or hit their walltime limit before finishing.
- `submit_train.sbatch`, the SLURM submission script (GPU count, walltime, partition, `--requeue` are all overridable at submit time without editing the file).
- `environment.yml`, a pinned conda environment for clusters that use Apptainer/Singularity + conda instead of Docker (the alternative to `infra/Dockerfile.train`).
- Weights & Biases logging setup (offline mode, since compute nodes are often not connected to the internet).

The important constraint here: this code has to be tested locally on tiny synthetic data before it's ever submitted to the cluster queue, so a shape mismatch doesn't waste an hour of queued GPU time.
