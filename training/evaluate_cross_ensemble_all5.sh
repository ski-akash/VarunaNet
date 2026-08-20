#!/bin/bash
#SBATCH --job-name=varunanet_eval_cross_ensemble_all5
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

# Logit-averaged cross-architecture ensemble of all 5 well-performing
# checkpoints on disk: the 3 primary U-Net seeds plus U-Net++ and
# SegFormer-B2. Compare against evaluate_cross_ensemble_top3.sh's
# 3-member version -- more diversity, but also 3 correlated U-Net votes.
python -m training.evaluate_cross_ensemble \
    --config-name=evaluate_cross_ensemble_all5 \
    device=cpu \
    2>&1 | tee "$TMPDIR/results/eval.log"

echo "Job completed at $(date)"
