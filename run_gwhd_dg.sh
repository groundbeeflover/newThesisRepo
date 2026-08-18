#!/usr/bin/env bash
set -euo pipefail

# RunPod: nondeterministic reproduction of Mattia's GWHD code --
# train_GWHD_dgfrcnn_mattia.py, his DGKarthik/GRL reproduction. Same
# conda-detection convention as run_gwhd_baseline.sh / run_gwhd_coral.sh.
#
# Ablation-style: two separate (LR, reg_weights) presets, each its own run
# directory, each run for 3 nondeterministic rounds:
#   lr1e-4_reg075  -- LR=1e-4, reg_weights 0.5 0.5 0.5 0.075 0.0001
#   lr1e-5_reg055  -- LR=1e-5, reg_weights 0.5 0.5 0.5 0.055 0.0001
#                     (Mattia's confirmed alpha_4, target avg test mAP@0.5 ~= 60.3)
# Only alpha_4 (the 4th reg weight) differs between presets; a/b/c/e keep
# this repo's existing 0.5/0.5/0.5/0.0001 convention in both.
#
# Each round is nondeterministic (cudnn.benchmark on, no --deterministic --
# the training script defaults to this when --deterministic is omitted).
# Nondeterministic mode doesn't need the PHYS_BATCH/ACCUM_STEPS split that
# the deterministic repro (run_repro_dgkarthik.sh) needs to fit BS=8 in
# memory, so batch_size=8 is passed straight through as a single physical
# batch.
#
# Each round still gets a distinct --seed (0/1/2) so weight init and data
# order vary too, not just conv-algorithm selection -- gives 3 genuinely
# independent trials per preset.
#
# Presets x rounds run SEQUENTIALLY (single RunPod GPU) -- 6 full
# train+eval passes total. Not detached: run inside tmux/screen so it
# survives an SSH disconnect:
#   tmux new -s gwhd_dg
#   bash run_gwhd_dg.sh
#   # Ctrl-b d to detach; `tmux attach -t gwhd_dg` to reattach later
#
# Safe to re-run / resume: each (preset, round) pair is skipped if its
# .done sentinel already exists.

ENV_NAME="DGOD"
ROUNDS=(0 1 2)

# tag:lr:alpha4  (reg_weights = 0.5 0.5 0.5 <alpha4> 0.0001)
HP_PRESETS=(
  "lr1e-4_reg075:1e-4:0.075"
  "lr1e-5_reg055:1e-5:0.055"
)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefer persistent conda if installed in /workspace; fall back to ~/miniconda3.
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${ENV_NAME}"

for PRESET in "${HP_PRESETS[@]}"; do
  IFS=':' read -r TAG LR ALPHA4 <<< "${PRESET}"
  REG_WEIGHTS=(0.5 0.5 0.5 "${ALPHA4}" 0.0001)

  RUN_NAME="gwhd_dgkarthik_nondet_${TAG}"
  RUN_DIR="runs/${RUN_NAME}"

  echo "Starting nondeterministic Mattia GWHD (DGKarthik/GRL) reproduction: preset=${TAG} lr=${LR} reg_weights=${REG_WEIGHTS[*]}, rounds: ${ROUNDS[*]}"
  echo "Batch size=8, single physical batch (no accumulation), nondeterministic (cudnn.benchmark)"

  mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

  cp train_GWHD_dgfrcnn_mattia.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
  cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
  pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
  conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
  nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
  git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

  for ROUND in "${ROUNDS[@]}"; do
    WEIGHTS_FILE="dgkarthik_nondet_${TAG}_round${ROUND}"

    if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ]; then
      echo "Preset ${TAG} round ${ROUND} already completed (found ${WEIGHTS_FILE}.done). Skipping."
      continue
    fi

    echo "=== mattia gwhd (dgkarthik) ${TAG} round ${ROUND}: training ==="
    # Own checkpoint + own training log per (preset, round).
    python train_GWHD_dgfrcnn_mattia.py \
      --exp dg \
      --weights_folder "${RUN_DIR}/checkpoints" \
      --weights_file "${WEIGHTS_FILE}" \
      --reg_weights "${REG_WEIGHTS[@]}" \
      --lr "${LR}" \
      --batch_size 8 \
      --accumulate_grad_batches 1 \
      --seed "${ROUND}" \
      2>&1 | tee "${RUN_DIR}/logs/train_round${ROUND}.log"

    echo "=== mattia gwhd (dgkarthik) ${TAG} round ${ROUND}: test evaluation ==="
    # Own test log + own test-results csv per (preset, round)
    # (${WEIGHTS_FILE}_test_map.csv, written into checkpoints/).
    python train_GWHD_dgfrcnn_mattia.py \
      --exp dg \
      --weights_folder "${RUN_DIR}/checkpoints" \
      --weights_file "${WEIGHTS_FILE}" \
      --eval_map \
      --eval_split test \
      2>&1 | tee "${RUN_DIR}/logs/test_round${ROUND}.log"

    touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
  done

  echo "Done (or already were) with preset ${TAG}. Check ${RUN_DIR}/checkpoints/*.done to confirm which rounds finished."
  echo "Per-round results: ${RUN_DIR}/checkpoints/dgkarthik_nondet_${TAG}_round*_test_map.csv"
  echo "Per-round per-domain results: ${RUN_DIR}/checkpoints/dgkarthik_nondet_${TAG}_round*_per_domain_map.csv"
done
