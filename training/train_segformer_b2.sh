#!/bin/bash
#SBATCH --job-name=varunanet_segformer_b2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-P100
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
export HF_HUB_OFFLINE=1

python -m training.train dataset=sen1floods11 model=segformer_b2 device=cuda epochs=30 seed=1 \
    early_stopping_patience=8 \
    wandb.mode=disabled \
    checkpoint_dir="$TMPDIR/results/checkpoints" \
    2>&1 | tee "$TMPDIR/results/train.log"

echo "Job completed at $(date)"
