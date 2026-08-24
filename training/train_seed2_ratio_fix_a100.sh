#!/bin/bash
#SBATCH --job-name=varunanet_train_s2_ratio_fix
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

# See training/train_seed1_ratio_fix_a100.sh for the full rationale -- same
# retrain, seed 2 (original: training/train_seed2_p100_v3.sh, job 1654,
# 0.6603 pooled IoU under the broken ratio channel).
python -m training.train dataset=sen1floods11 device=cuda epochs=30 seed=2 \
    early_stopping_patience=8 \
    wandb.mode=disabled \
    checkpoint_dir="$TMPDIR/results/checkpoints" \
    2>&1 | tee "$TMPDIR/results/train.log"

echo "Job completed at $(date)"
