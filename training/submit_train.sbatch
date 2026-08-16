#!/bin/bash
# SLURM submission script for training on the cluster (spec section 8:
# "sbatch submission scripts with configurable walltime, partition, GPU
# count, and --requeue").
#
# "Configurable" here means what it normally means for sbatch: every
# #SBATCH directive below is a *default*, not a hardcoded requirement --
# any of them can be overridden per-submission without editing this file,
# since sbatch command-line flags win over the directives in the script:
#
#   sbatch --time=8:00:00 --partition=gpu-short --gres=gpu:2 \
#       training/submit_train.sbatch
#
# Extra arguments after the script path are forwarded to
# `python -m training.train` as Hydra overrides, e.g.:
#
#   sbatch training/submit_train.sbatch epochs=100 optimizer.lr=3e-4
#
# --requeue + automatic resume: if this job is preempted or hits its
# walltime, SLURM (given a partition/QOS that supports it) requeues it as
# a fresh run of this same script from the top. No special "find my last
# checkpoint" logic lives here on purpose -- training/train.py's main()
# already checks checkpoint_dir for an existing checkpoint and resumes
# from it automatically (training/checkpoint.py's find_latest_checkpoint)
# before training/train.py ever proceeds. That keeps this script a plain,
# boring "run the same command again" and puts the actually-important
# correctness logic (the one this project got wrong once already with
# itertools.cycle, and fixed) in the part that's covered by tests.

#SBATCH --job-name=varunanet-train
#SBATCH --partition=gpu           # placeholder -- set to your cluster's actual GPU partition name
#SBATCH --gres=gpu:1              # GPU count; this project targets a single A100 (spec section 13)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00           # placeholder -- walltime limit wasn't confirmed yet; adjust to your cluster's actual limit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue

set -euo pipefail

mkdir -p logs

# Prefer the conda environment (training/environment.yml) if it's been
# created; fall back to an Apptainer/Singularity image built from
# infra/Dockerfile.train for clusters that don't offer conda directly
# (spec section 8: "many clusters use Singularity/Apptainer rather than
# Docker"). Exactly one of these two branches is expected to apply on any
# given cluster -- adjust to match how this was actually set up there.
if command -v conda >/dev/null 2>&1 && conda env list | grep -q "^varunanet "; then
    eval "$(conda shell.bash hook)"
    conda activate varunanet
    RUN_CMD="python -m training.train"
elif [ -f "varunanet.sif" ]; then
    RUN_CMD="apptainer exec --nv varunanet.sif python -m training.train"
else
    echo "Neither a 'varunanet' conda env nor varunanet.sif was found." >&2
    echo "Create one from training/environment.yml, or build" >&2
    echo "varunanet.sif from infra/Dockerfile.train (e.g. via" >&2
    echo "'apptainer build varunanet.sif docker-daemon://varunanet-train:latest')." >&2
    exit 1
fi

echo "job ${SLURM_JOB_ID:-local}, node $(hostname), starting: ${RUN_CMD} $*"

# srun (rather than running python directly) so SLURM's job accounting
# and signal propagation on preemption apply to the actual training
# process, not just this wrapper script.
srun ${RUN_CMD} \
    checkpoint_dir="checkpoints/${SLURM_JOB_NAME:-varunanet}" \
    wandb.dir="wandb_runs/${SLURM_JOB_NAME:-varunanet}" \
    "$@"
