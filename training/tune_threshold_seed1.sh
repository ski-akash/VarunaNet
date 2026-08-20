#!/bin/bash
#SBATCH --job-name=varunanet_tune_threshold_s1
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

# Sweeps the sigmoid threshold on the *val* split (never test) for the
# primary U-Net's seed=1 checkpoint -- zero retraining, just a decision
# rule change. See training/tune_threshold.py's own docstring for why
# val, not test.
python -m training.tune_threshold \
    checkpoint=/userhome/mtech/akashc1005/job_results/1634/checkpoints/best.pt \
    dataset=sen1floods11 \
    device=cpu \
    2>&1 | tee "$TMPDIR/results/tune.log"

echo "Job completed at $(date)"
