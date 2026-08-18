#!/bin/bash
#SBATCH --job-name=varunanet_eval_segformer_b0
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

cd "$TMPDIR" || exit 1
mkdir -p ./results

echo "Job $SLURM_JOB_ID running on $(hostname)"

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

export HF_HUB_OFFLINE=1

python -m training.evaluate_test \
    checkpoint=/userhome/mtech/akashc1005/job_results/1670/checkpoints/best.pt \
    model=segformer_b0 \
    dataset=sen1floods11 device=cpu \
    2>&1 | tee "$TMPDIR/results/eval.log"

echo "Job completed at $(date)"
