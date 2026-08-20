#!/bin/bash
#SBATCH --job-name=varunanet_eval_cross_ensemble_top3
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

# Logit-averaged cross-architecture ensemble of this project's 3
# strongest single checkpoints (SegFormer-B2, U-Net++, primary U-Net
# seed=1) -- zero retraining, extending the same-architecture ensemble
# win (evaluate_ensemble_3seed.sh, 0.4303) across architecture families.
python -m training.evaluate_cross_ensemble \
    --config-name=evaluate_cross_ensemble_top3 \
    device=cpu \
    2>&1 | tee "$TMPDIR/results/eval.log"

echo "Job completed at $(date)"
