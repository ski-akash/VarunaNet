#!/bin/bash
#SBATCH --job-name=varunanet_train_s1_ratio_fix
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

# Exact same recipe as the original primary U-Net run (training/train_seed1_p100_v3.sh,
# job 1634 -- 0.6761 pooled test IoU under the broken ratio channel). Only the data
# pipeline changed: VV_VH_ratio is now VV_db - VH_db (subtraction, physically correct
# for two already-logarithmic quantities) instead of VV_db / VH_db, which blew up to
# extreme outliers whenever VH_db landed near zero. Every existing 5-channel checkpoint
# was trained on that broken channel and normalization_stats.json now reflects the fixed
# one, so re-evaluating old checkpoints against it is a real train/serve mismatch --
# this run is what actually answers whether the fix helps the CNN, which the tree-based
# baselines (ExtraTrees, unaffected: 0.6425 -> 0.6427 pooled IoU, within noise) couldn't.
python -m training.train dataset=sen1floods11 device=cuda epochs=30 seed=1 \
    early_stopping_patience=8 \
    wandb.mode=disabled \
    checkpoint_dir="$TMPDIR/results/checkpoints" \
    2>&1 | tee "$TMPDIR/results/train.log"

echo "Job completed at $(date)"
