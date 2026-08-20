#!/bin/bash
#SBATCH --job-name=varunanet_eval_cross_ensemble_top3_tta
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

# Same top-3 cross-architecture ensemble as evaluate_cross_ensemble_top3.sh
# (0.4427, this project's current best), but with each member's own
# 4-way flip TTA folded in before combining -- TTA helped 2 of these 3
# architectures individually (cnn_results.md), worth checking whether
# that compounds with cross-architecture ensembling or not.
python -m training.evaluate_cross_ensemble \
    --config-name=evaluate_cross_ensemble_top3 \
    device=cpu \
    tta=true \
    2>&1 | tee "$TMPDIR/results/eval.log"

echo "Job completed at $(date)"
