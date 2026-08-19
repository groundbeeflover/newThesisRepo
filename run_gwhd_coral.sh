#!/usr/bin/env bash
set -euo pipefail

# runpod, nondeterministic coral runs for the two sampler configs i care about,
# domcap2_bs8 (domain_capped, k=2) and diverse_bs8 (domain_diverse)
#
# this replaces the old generic
# `run_gwhd_coral.sh <tag> <sampler> <batch_size> <domains_per_batch> [lr] [wada]`
# interface that run_coral_sampler_ablation.sh used for its full 5 point bs2/bs8
# grid, that grid isn't reproduced here, only the two bs8 configs below, so if
# the ablation script's other points (capped2_bs2, natural_bs8 and so on) are
# needed again they'll want their own runner
#
# each sampler config runs under two (lr, reg_weights) presets, each with its own
# run directory, each run 3 times:
#   lr1e-4_reg075  -- lr=1e-4, reg_weights 0.5 0.5 0.5 0.075 0.0001
#   lr1e-5_reg055  -- lr=1e-5, reg_weights 0.5 0.5 0.5 0.055 0.0001, which lines
#                     up with run_repro_coral.sh and the baseline/dgkarthik
#                     deterministic repros
# only alpha_4, the 4th one, changes between the presets, a/b/c/e stay on this
# repo's usual 0.5/0.5/0.5/0.0001 in both
#
# 2 configs x 2 presets = 4 run directories, 3 rounds each, so 12 full train+eval
# passes
#
# every round is nondeterministic, cudnn.benchmark on and no --deterministic,
# which is the training script's default, so no need for the
# PHYS_BATCH/ACCUM_STEPS split the deterministic repro (run_repro_coral.sh) needs
# to fit bs=8 in memory, batch_size=8 goes through as one real batch
#
# each round still gets its own --seed (0/1/2) so weight init, data order and the
# sampler draw all move around, not just conv algorithm choice, making them 3
# properly independent trials per (config, preset)
#
# configs x presets x rounds run one after the other, single runpod gpu, nothing
# detaches this, run it in tmux or screen so an ssh drop doesn't kill it:
#   tmux new -s gwhd_coral
#   bash run_gwhd_coral.sh
#   #ctrl-b d to detach, tmux attach -t gwhd_coral to come back
#
# fine to re-run, any (config, preset, round) with a .done sentinel gets skipped

ENV_NAME="DGOD"
ROUNDS=(0 1 2)

#tag:sampler:domains_per_batch
CONFIGS=(
  "domcap2_bs8:domain_capped:2"
  "diverse_bs8:domain_diverse:8"
)

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

for CONFIG in "${CONFIGS[@]}"; do
  IFS=':' read -r CONFIG_TAG SAMPLER DOMAINS_PER_BATCH <<< "${CONFIG}"

  for PRESET in "${HP_PRESETS[@]}"; do
    IFS=':' read -r HP_TAG LR ALPHA4 <<< "${PRESET}"
    REG_WEIGHTS=(0.5 0.5 0.5 "${ALPHA4}" 0.0001)

    TAG="${CONFIG_TAG}_${HP_TAG}"
    RUN_NAME="gwhd_coral_nondet_${TAG}"
    RUN_DIR="runs/${RUN_NAME}"
    CKPT_DIR="${RUN_DIR}/checkpoints"

    echo "Starting nondeterministic CORAL run: config=${CONFIG_TAG} (sampler=${SAMPLER} domains_per_batch=${DOMAINS_PER_BATCH}) preset=${HP_TAG} (lr=${LR} reg_weights=${REG_WEIGHTS[*]}), rounds: ${ROUNDS[*]}"
    echo "Batch size=8, single physical batch (no accumulation), nondeterministic (cudnn.benchmark)"

    mkdir -p "${CKPT_DIR}" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

    cp train_GWHD_coralfrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
    cp -r dg "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
    cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
    pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
    conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
    nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
    git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

    for ROUND in "${ROUNDS[@]}"; do
      WEIGHTS_FILE="coral_nondet_${TAG}_round${ROUND}"

      if [ -f "${CKPT_DIR}/${WEIGHTS_FILE}.done" ]; then
        echo "Config ${CONFIG_TAG} preset ${HP_TAG} round ${ROUND} already completed (found ${WEIGHTS_FILE}.done). Skipping."
        continue
      fi

      echo "=== coral ${TAG} round ${ROUND}: training ==="
      #its own checkpoint and its own training log per (config, preset, round)
      python train_GWHD_coralfrcnn.py \
        --exp coral \
        --weights_folder "${CKPT_DIR}" \
        --weights_file "${WEIGHTS_FILE}" \
        --sampler "${SAMPLER}" \
        --domains_per_batch "${DOMAINS_PER_BATCH}" \
        --reg_weights "${REG_WEIGHTS[@]}" \
        --lr "${LR}" \
        --batch_size 8 \
        --accumulate_grad_batches 1 \
        --num_workers 16 \
        --seed "${ROUND}" \
        2>&1 | tee "${RUN_DIR}/logs/train_round${ROUND}.log"

      echo "=== coral ${TAG} round ${ROUND}: test evaluation ==="
      #its own test log and its own test results csv per (config, preset,
      #round), that's ${WEIGHTS_FILE}_test_map.csv, dropped into checkpoints/
      python train_GWHD_coralfrcnn.py \
        --exp coral \
        --weights_folder "${CKPT_DIR}" \
        --weights_file "${WEIGHTS_FILE}" \
        --eval_map \
        --eval_split test \
        2>&1 | tee "${RUN_DIR}/logs/test_round${ROUND}.log"

      touch "${CKPT_DIR}/${WEIGHTS_FILE}.done"
    done

    echo "Done (or already were) with ${TAG}. Check ${CKPT_DIR}/*.done to confirm which rounds finished."
    echo "Per-round results: ${CKPT_DIR}/coral_nondet_${TAG}_round*_test_map.csv"
    echo "Per-round per-domain results: ${CKPT_DIR}/coral_nondet_${TAG}_round*_per_domain_map.csv"
  done
done
