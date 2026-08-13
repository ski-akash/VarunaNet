# training/

The actual training loop, plus everything needed to run it on a remote SLURM GPU cluster instead of a laptop.

This folder holds:
- The config-driven training loop (Hydra/YAML configs — no hardcoded paths or hyperparameters, so the same code runs locally on a tiny test tensor or on the cluster on the full dataset).
- Checkpointing and resume logic, since cluster jobs can get preempted or hit their walltime limit before finishing.
- `sbatch` submission scripts for SLURM (GPU count, walltime, partition, requeue behavior).
- Weights & Biases logging setup (offline mode, since compute nodes are often not connected to the internet).

The important constraint here: this code has to be tested locally on tiny synthetic data before it's ever submitted to the cluster queue, so a shape mismatch doesn't waste an hour of queued GPU time.
