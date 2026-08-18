#!/bin/bash
#SBATCH --job-name=varunanet_per_event
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

cd "$TMPDIR" || exit 1
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python -m training.per_event_eval \
    checkpoint=/userhome/mtech/akashc1005/job_results/1634/checkpoints/best.pt \
    dataset=sen1floods11 device=cpu

echo "Job completed at $(date)"
