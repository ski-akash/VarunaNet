#!/bin/bash
#SBATCH --job-name=varunanet_speckle_looks1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$TMPDIR" || exit 1
mkdir -p ./results

echo "Job $SLURM_JOB_ID running on $(hostname)"

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

nvidia-smi | tee "$TMPDIR/results/nvidia-smi.txt"

export WANDB_MODE=offline

# Speckle-noise augmentation sweep, strong noise (looks=1 -- highest
# variance in the Gamma(shape=looks, scale=1/looks) model, see
# data/speckle.py). Same recipe as the real baseline (seed=1, 30 epochs,
# patience=8) otherwise, so this is a clean A/B against the existing
# mean_iou=0.4198 test-split result.
python -m training.train dataset=sen1floods11 dataset.speckle_prob=0.5 dataset.speckle_looks=1.0 \
    device=cuda epochs=30 seed=1 early_stopping_patience=8 \
    wandb.mode=disabled \
    checkpoint_dir="$TMPDIR/results/checkpoints" \
    2>&1 | tee "$TMPDIR/results/train.log"

echo "Job completed at $(date)"
