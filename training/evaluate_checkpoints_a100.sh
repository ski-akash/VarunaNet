#!/bin/bash
#SBATCH --job-name=varunanet_evaluate_checkpoints
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$TMPDIR" || exit 1
mkdir -p ./results

echo "Job $SLURM_JOB_ID running on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python -m training.evaluate_checkpoints \
    2>&1 | tee "$TMPDIR/results/evaluate_checkpoints.log"

echo "Job completed at $(date)"
