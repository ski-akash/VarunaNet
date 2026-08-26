#!/bin/bash
#SBATCH --job-name=varunanet_change_aware_s3_ratio_fix
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

# Retrain after the VV_VH_ratio fix (see training/train_seed1_ratio_fix_a100.sh
# for the full rationale). Original run: training/train_change_aware_s3_a100.sh, job 1715,
# 0.4098 mean IoU under the broken ratio channel + stale normalization stats.
# Same recipe, only the (already-fixed) data pipeline is different.
python -m training.train dataset=sen1floods11_change_aware model=change_aware_unet epochs=30 seed=3 early_stopping_patience=8 device=cuda \
    wandb.mode=disabled \
    checkpoint_dir="$TMPDIR/results/checkpoints" \
    2>&1 | tee "$TMPDIR/results/train.log"

echo "Job completed at $(date)"
