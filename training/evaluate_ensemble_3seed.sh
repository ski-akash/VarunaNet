#!/bin/bash
#SBATCH --job-name=varunanet_eval_ensemble_3seed
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

# Logit-averaged ensemble of the primary U-Net's 3 real trained seeds
# (checkpoints from jobs 2062/2063/2065, the ratio-fix retrains of
# 1634/1654/1660) -- zero retraining, just
# combining what's already on disk.
python -m training.evaluate_ensemble \
    checkpoints=[/userhome/mtech/akashc1005/job_results/2062/checkpoints/best.pt,/userhome/mtech/akashc1005/job_results/2063/checkpoints/best.pt,/userhome/mtech/akashc1005/job_results/2065/checkpoints/best.pt] \
    dataset=sen1floods11 \
    device=cpu \
    2>&1 | tee "$TMPDIR/results/eval.log"

echo "Job completed at $(date)"
