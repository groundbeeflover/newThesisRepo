#!/usr/bin/env bash
set -euo pipefail

# runpod, nondeterministic version of mattia's pure baseline, so
# train_GWHD_baseline_clean.py, stock torchvision faster r-cnn, no da heads and
# no fasterrcnn.py wrapper, which is why there's no --reg_weights here, there's
# no da loss to weight in the first place
#
# two lr presets, each with its own run directory, each run 3 times:
#   lr1e-4  -- lr=1e-4
#   lr1e-5  -- lr=1e-5, mattia's setting, target avg test map@0.5 around 54.7
#
# every round is nondeterministic, cudnn.benchmark on and no --deterministic,
# which is what the training script does by default, that means i don't need the
# PHYS_BATCH/ACCUM_STEPS split the deterministic repro (run_repro_baseline.sh)
# needs to squeeze bs=8 into memory, since cudnn.benchmark picks whichever conv
# algorithms actually fit instead of being forced onto the slow memory hungry
# deterministic ones, so batch_size=8 goes straight through as one real batch
#
# each round still gets its own --seed (0/1/2) so weight init and data order move
# around too, not just which conv algorithm got picked, which makes them 3
# properly independent trials per preset
#
# presets x rounds run one after the other since it's a single runpod gpu, so 6
# full train+eval passes, nothing detaches this, run it in tmux or screen so an
# ssh drop doesn't kill it:
#   tmux new -s gwhd_baseline
#   bash run_gwhd_baseline.sh
#   #ctrl-b d to detach, tmux attach -t gwhd_baseline to come back
#
# fine to re-run, any (preset, round) with a .done sentinel gets skipped

ENV_NAME="DGOD"
ROUNDS=(0 1 2)

#tag:lr
HP_PRESETS=(
  "lr1e-4:1e-4"
  "lr1e-5:1e-5"
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
  IFS=':' read -r TAG LR <<< "${PRESET}"

  RUN_NAME="gwhd_baseline_nondet_${TAG}"
  RUN_DIR="runs/${RUN_NAME}"

  echo "Starting nondeterministic pure-baseline reproduction: preset=${TAG} lr=${LR}, rounds: ${ROUNDS[*]}"
  echo "Batch size=8, single physical batch (no accumulation), nondeterministic (cudnn.benchmark)"

  mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

  cp train_GWHD_baseline_clean.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
  cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
  pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
  conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
  nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
  git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

  for ROUND in "${ROUNDS[@]}"; do
    WEIGHTS_FILE="baseline_nondet_${TAG}_round${ROUND}"

    if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ]; then
      echo "Preset ${TAG} round ${ROUND} already completed (found ${WEIGHTS_FILE}.done). Skipping."
      continue
    fi

    echo "=== pure baseline ${TAG} round ${ROUND}: training ==="
    #its own checkpoint and its own training log per (preset, round)
    python train_GWHD_baseline_clean.py \
      --weights_folder "${RUN_DIR}/checkpoints" \
      --weights_file "${WEIGHTS_FILE}" \
      --lr "${LR}" \
      --batch_size 8 \
      --accumulate_grad_batches 1 \
      --num_workers 16 \
      --seed "${ROUND}" \
      2>&1 | tee "${RUN_DIR}/logs/train_round${ROUND}.log"

    echo "=== pure baseline ${TAG} round ${ROUND}: test evaluation ==="
    #its own test log and its own test results csv per (preset, round),
    #that's ${WEIGHTS_FILE}_test_map.csv, dropped into checkpoints/
    python train_GWHD_baseline_clean.py \
      --weights_folder "${RUN_DIR}/checkpoints" \
      --weights_file "${WEIGHTS_FILE}" \
      --eval_map \
      --eval_split test \
      2>&1 | tee "${RUN_DIR}/logs/test_round${ROUND}.log"

    touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
  done

  echo "Done (or already were) with preset ${TAG}. Check ${RUN_DIR}/checkpoints/*.done to confirm which rounds finished."
  echo "Per-round results: ${RUN_DIR}/checkpoints/baseline_nondet_${TAG}_round*_test_map.csv"
  echo "Per-round per-domain results: ${RUN_DIR}/checkpoints/baseline_nondet_${TAG}_round*_per_domain_map.csv"
done
