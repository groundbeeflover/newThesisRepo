#!/usr/bin/env bash
set -euo pipefail

# Optional exploratory ablation: does rebalancing the CORAL branch weights
# (alpha_1..5 in Eq. 2 of the thesis) toward equal per-branch influence
# change anything, versus the current fixed weights?
#
# Background (see writtenWork/coral_alpha_rebalance_experiment.md for the
# full writeup): pulling and analysing the TensorBoard logs from all CORAL
# training runs to date showed the five auxiliary branches sit at very
# different raw magnitudes -- L_img ~1e-2, L_ins ~4e-4, L_cst ~3e-5,
# L_ds_adv ~7e-4, L_ds_cls ~3e-4 (means across 22 alpha-tuned runs). Under
# the CURRENT weights (0.5, 0.5, 0.5, 0.075, 0.0001), that magnitude gap
# means L_img dominates the total auxiliary loss for most of training,
# while L_ds_cls's weighted contribution is ~4-5 orders of magnitude
# smaller than every other branch -- i.e. functionally zero, despite being
# one of five nominal loss terms.
#
# This script does NOT do a real hyperparameter sweep (no time for that
# before the deadline). It runs exactly TWO single-seed configs, same
# sampler/batch-size/LR as the thesis's best-known CORAL config
# (domain_diverse, batch_size=8, lr=1e-4, i.e. "diverse_bs8"), differing
# ONLY in reg_weights:
#
#   current    : 0.5 0.5 0.5 0.075  0.0001   <- what's in the thesis now
#   rebalanced : 0.5 0.5 0.5 0.32   0.64     <- see derivation below
#
# Rebalanced weights are derived by targeting each of L_ds_adv and
# L_ds_cls's WEIGHTED contribution to match L_ins's current weighted
# contribution (k = alpha2 * mean(raw_ins) ~= 2.19e-4 across the 22 tuned
# runs), i.e. new_alpha_i = k / mean(raw_i):
#   alpha_4 (ds_adv) = 2.19e-4 / 6.79e-4 ~= 0.32
#   alpha_5 (ds_cls) = 2.19e-4 / 3.41e-4 ~= 0.64
# alpha_1/2/3 (img/ins/cst) are left untouched -- deliberately scoped down
# to just the two concept-shift branches the findings flagged as
# underweighted, rather than also re-balancing img/ins/cst against each
# other (a separate, less clear-cut question; see the markdown writeup).
#
# This is intentionally the SAME evidentiary bar as the existing sampler
# ablation (single seed, small grid, explicitly reported as preliminary) --
# not a claim of a tuned optimum.
#
# Usage:
#   ./run_coral_alpha_rebalance.sh [seed]
# seed defaults to 42 (matches the historical diverse_bs8 result reported
# in the thesis, so the "current" run here is a same-seed sanity check of
# that number, not just a re-explanation of it).
#
# Run inside tmux/screen so it survives an SSH disconnect:
#   tmux new -s coral_alpha
#   bash run_coral_alpha_rebalance.sh
#   # Ctrl-b d to detach; `tmux attach -t coral_alpha` to reattach later
#
# Cost: 2 runs at bs8/diverse_bs8, ~13-19 epochs each based on historical
# runs of this exact config -- roughly the same wall-clock as 2 points in
# the existing sampler ablation grid (a few hours total on a single GPU,
# not an overnight job).

SEED="${1:-42}"
ENV_NAME="DGOD"
RESULTS_DIR="coral_alpha_rebalance_runs"
mkdir -p "${RESULTS_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${ENV_NAME}"

# tag:reg_weights (space-separated, matches --reg_weights nargs=5)
CONFIGS=(
  "current:0.5 0.5 0.5 0.075 0.0001"
  "rebalanced:0.5 0.5 0.5 0.32 0.64"
)

for CONFIG in "${CONFIGS[@]}"; do
  IFS=':' read -r TAG REG_WEIGHTS_STR <<< "${CONFIG}"
  read -ra REG_WEIGHTS <<< "${REG_WEIGHTS_STR}"

  WEIGHTS_FILE="alpha_rebalance_${TAG}_seed${SEED}"
  CKPT_DIR="${RESULTS_DIR}/checkpoints"
  SNAPSHOT_DIR="${RESULTS_DIR}/config_snapshot_${TAG}"
  mkdir -p "${CKPT_DIR}" "${SNAPSHOT_DIR}"

  if [ -f "${CKPT_DIR}/${WEIGHTS_FILE}.done" ]; then
    echo "=== ${TAG} (seed ${SEED}) already completed (found ${WEIGHTS_FILE}.done). Skipping. ==="
    continue
  fi

  echo "=== [${TAG}] reg_weights=${REG_WEIGHTS[*]} seed=${SEED}: config snapshot ==="
  cp train_GWHD_coralfrcnn.py "${SNAPSHOT_DIR}/" 2>/dev/null || true
  cp -r dg "${SNAPSHOT_DIR}/" 2>/dev/null || true
  git rev-parse HEAD > "${SNAPSHOT_DIR}/git_commit.txt" 2>/dev/null || true
  echo "${REG_WEIGHTS[*]}" > "${SNAPSHOT_DIR}/reg_weights.txt"

  echo "=== [${TAG}] training (diverse_bs8: sampler=domain_diverse, batch_size=8, lr=1e-4) ==="
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --weights_folder "${CKPT_DIR}" \
    --weights_file "${WEIGHTS_FILE}" \
    --sampler domain_diverse \
    --batch_size 8 \
    --lr 0.0001 \
    --reg_weights "${REG_WEIGHTS[@]}" \
    --num_workers 16 \
    --seed "${SEED}" \
    2>&1 | tee "${RESULTS_DIR}/${WEIGHTS_FILE}_train.log"

  echo "=== [${TAG}] test evaluation ==="
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --weights_folder "${CKPT_DIR}" \
    --weights_file "${WEIGHTS_FILE}" \
    --eval_map \
    --eval_split test \
    2>&1 | tee "${RESULTS_DIR}/${WEIGHTS_FILE}_test_eval.log"

  touch "${CKPT_DIR}/${WEIGHTS_FILE}.done"
  echo "=== [${TAG}] done. Results: ${CKPT_DIR}/${WEIGHTS_FILE}_test_map.csv ==="
done

echo
echo "Both configs finished (or already were). Compare:"
echo "  ${RESULTS_DIR}/checkpoints/alpha_rebalance_current_seed${SEED}_test_map.csv"
echo "  ${RESULTS_DIR}/checkpoints/alpha_rebalance_rebalanced_seed${SEED}_test_map.csv"
echo "Also worth pulling the TensorBoard logs for these two runs afterward (same"
echo "method as before -- see writtenWork/coral_branch_logging_findings.md section 2)"
echo "to check whether the rebalanced run's weighted branches actually landed"
echo "closer to parity, not just whether test mAP moved."
