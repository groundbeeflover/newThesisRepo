#!/usr/bin/env bash
set -euo pipefail

# optional exploratory ablation: does rebalancing the coral branch weights
# (alpha_1..5 in equation 2 of the thesis) towards equal per branch influence
# actually change anything, compared to the current fixed weights?
#
# background, and writtenWork/coral_alpha_rebalance_experiment.md has the full
# writeup: pulling the tensorboard logs from every coral run so far showed the
# five auxiliary branches sitting at wildly different raw magnitudes, l_img
# around 1e-2, l_ins around 4e-4, l_cst around 3e-5, l_ds_adv around 7e-4 and
# l_ds_cls around 3e-4, averaged over 22 alpha tuned runs, with the current
# weights (0.5, 0.5, 0.5, 0.075, 0.0001) that gap means l_img dominates the total
# auxiliary loss for most of training, while l_ds_cls's weighted contribution
# sits 4 to 5 orders of magnitude below every other branch, so it's doing
# nothing at all despite being one of the five nominal loss terms
#
# this is not a real hyperparameter sweep, there's no time for one before the
# deadline, it runs exactly two single seed configs on the same
# sampler/batch size/lr as the thesis's best known coral config
# (domain_diverse, batch_size=8, lr=1e-4, so "diverse_bs8"), and the only thing
# that differs is reg_weights:
#
#   current    : 0.5 0.5 0.5 0.075  0.0001   <- what's in the thesis right now
#   rebalanced : 0.5 0.5 0.5 0.32   0.64     <- derivation below
#
# the rebalanced weights come from making l_ds_adv and l_ds_cls's weighted
# contributions match l_ins's current weighted contribution
# (k = alpha2 * mean(raw_ins), about 2.19e-4 across the 22 tuned runs), so
# new_alpha_i = k / mean(raw_i):
#   alpha_4 (ds_adv) = 2.19e-4 / 6.79e-4, about 0.32
#   alpha_5 (ds_cls) = 2.19e-4 / 3.41e-4, about 0.64
# alpha_1/2/3 (img/ins/cst) are left alone deliberately, scoping this down to
# just the two concept shift branches the findings flagged as underweighted
# rather than also rebalancing img/ins/cst against each other, which is a
# separate and much less clear cut question (again, see the markdown writeup)
#
# this is on purpose held to the same evidentiary bar as the existing sampler
# ablation, single seed, small grid, reported as preliminary, it is not a claim
# of a tuned optimum
#
# usage:
#   ./run_coral_alpha_rebalance.sh [seed]
# seed defaults to 42, which matches the historical diverse_bs8 result in the
# thesis, so the "current" run here doubles as a same seed sanity check on that
# number rather than just re-explaining it
#
# run inside tmux or screen so an ssh drop doesn't kill it:
#   tmux new -s coral_alpha
#   bash run_coral_alpha_rebalance.sh
#   #ctrl-b d to detach, tmux attach -t coral_alpha to come back
#
# cost: 2 runs at diverse_bs8, 13 to 19 epochs each going off historical runs of
# this exact config, so about the same wall clock as 2 points in the existing
# sampler ablation grid, a few hours on one gpu, not an overnight job

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

#tag:reg_weights, space separated to match --reg_weights nargs=5
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
