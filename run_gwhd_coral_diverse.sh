#!/usr/bin/env bash
set -euo pipefail

# runpod, the diverse_bs8 (domain_diverse) coral runs only
#
# this is run_gwhd_coral.sh with the CONFIGS list cut down to the one sampler,
# for picking the diverse arm back up after it was stopped part way through,
# without the domcap2 runs being walked over again. everything else -- presets,
# seeds, batch size, snapshotting, sentinels -- is identical to that script, so
# runs started by either one are interchangeable
#
# domain_diverse puts one image from each of 8 distinct domains in every batch,
# against domain_capped(2) which puts 4 images each from 2 domains. so the
# covariances behind the coral loss come from many thin per domain groups rather
# than two fat ones, which is the tradeoff dg/samplers.py is about
#
# two (lr, reg_weights) presets, each with its own run directory, each run 3
# times:
#   lr1e-4_reg075  -- lr=1e-4, reg_weights 0.5 0.5 0.5 0.075 0.0001
#   lr1e-5_reg055  -- lr=1e-5, reg_weights 0.5 0.5 0.5 0.055 0.0001
# only alpha_4 changes between them
#
# resuming: a round that was interrupted mid training picks up on its own, the
# training script looks for <weights_file>-last.ckpt in the checkpoints folder
# and passes it to trainer.fit as ckpt_path, so the partly trained round 0 in
# runs/gwhd_coral_nondet_diverse_bs8_lr1e-4_reg075 continues from its last
# completed epoch rather than restarting. a round that finished has a .done
# sentinel and is skipped outright, so this is safe to re-run at any point
#
# nothing detaches this, one gpu, runs serially. use tmux so an ssh drop does
# not kill it again:
#   tmux new -s gwhd_coral_diverse
#   bash run_gwhd_coral_diverse.sh
#   #ctrl-b d to detach, tmux attach -t gwhd_coral_diverse to come back
#
# env overrides
#   ENV_NAME   conda env                       (default DGOD)
#   ROUNDS     space separated round numbers   (default "0 1 2")
#   PRESETS    space separated preset tags     (default both, see HP_PRESETS)
#              e.g. PRESETS="lr1e-4_reg075" to finish the interrupted one first

ENV_NAME="${ENV_NAME:-DGOD}"
read -r -a ROUNDS <<< "${ROUNDS:-0 1 2}"

#tag:sampler:domains_per_batch
CONFIGS=(
  "diverse_bs8:domain_diverse:8"
)

#tag:lr:alpha4, reg_weights ends up 0.5 0.5 0.5 <alpha4> 0.0001
ALL_PRESETS=(
  "lr1e-4_reg075:1e-4:0.075"
  "lr1e-5_reg055:1e-5:0.055"
)

#PRESETS filters ALL_PRESETS by tag, empty means keep them all
HP_PRESETS=()
if [ -z "${PRESETS:-}" ]; then
  HP_PRESETS=("${ALL_PRESETS[@]}")
else
  for WANT in ${PRESETS}; do
    for P in "${ALL_PRESETS[@]}"; do
      [ "${P%%:*}" = "${WANT}" ] && HP_PRESETS+=("${P}")
    done
  done
  if [ ${#HP_PRESETS[@]} -eq 0 ]; then
    echo "!! PRESETS='${PRESETS}' matched none of: ${ALL_PRESETS[*]%%:*}" >&2
    exit 1
  fi
fi

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

      if [ -f "${CKPT_DIR}/${WEIGHTS_FILE}-last.ckpt" ]; then
        echo "Found ${WEIGHTS_FILE}-last.ckpt, this round will resume from it rather than restart."
      fi

      echo "=== coral ${TAG} round ${ROUND}: training ==="
      #its own checkpoint and its own training log per (config, preset, round).
      #the train log is appended to, not truncated, so a resumed round keeps the
      #epochs it already did instead of the tee wiping them
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
        2>&1 | tee -a "${RUN_DIR}/logs/train_round${ROUND}.log"

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
