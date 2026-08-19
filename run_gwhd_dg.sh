#!/usr/bin/env bash
set -euo pipefail

# runpod, nondeterministic version of mattia's gwhd code, so
# train_GWHD_dgfrcnn_mattia.py, his dgkarthik/grl reproduction
#
# two (lr, reg_weights) presets, each with its own run directory, each run 3
# times:
#   lr1e-4_reg075  -- lr=1e-4, reg_weights 0.5 0.5 0.5 0.075 0.0001
#   lr1e-5_reg055  -- lr=1e-5, reg_weights 0.5 0.5 0.5 0.055 0.0001, mattia's
#                     confirmed alpha_4, target avg test map@0.5 around 60.3
# only alpha_4, the 4th one, changes between the presets, a/b/c/e stay on this
# repo's usual 0.5/0.5/0.5/0.0001 in both
#
# every round is nondeterministic, cudnn.benchmark on and no --deterministic,
# which is the training script's default, so no need for the
# PHYS_BATCH/ACCUM_STEPS split the deterministic repro (run_repro_dgkarthik.sh)
# needs to fit bs=8 in memory, batch_size=8 goes through as one real batch
#
# each round still gets its own --seed (0/1/2) so weight init and data order move
# around too, not just conv algorithm choice, making them 3 properly independent
# trials per preset
#
# presets x rounds run one after the other since it's a single runpod gpu, so 6
# full train+eval passes, nothing detaches this, run it in tmux or screen so an
# ssh drop doesn't kill it:
#   tmux new -s gwhd_dg
#   bash run_gwhd_dg.sh
#   #ctrl-b d to detach, tmux attach -t gwhd_dg to come back
#
# fine to re-run, any (preset, round) with a .done sentinel gets skipped

ENV_NAME="DGOD"
ROUNDS=(0 1 2)

#tag:lr:alpha4, reg_weights ends up 0.5 0.5 0.5 <alpha4> 0.0001
HP_PRESETS=(
  "lr1e-4_reg075:1e-4:0.075"
  "lr1e-5_reg055:1e-5:0.055"
)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#use the conda in /workspace if it's there, it survives a pod restart,
#otherwise fall back to the one in home
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
    #its own checkpoint and its own training log per (preset, round)
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
    #its own test log and its own test results csv per (preset, round),
    #that's ${WEIGHTS_FILE}_test_map.csv, dropped into checkpoints/
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
