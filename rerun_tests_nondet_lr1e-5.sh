#!/usr/bin/env bash
# runpod. re-runs the test split eval for the two lr1e-5 nondet runs off the
# checkpoints already in runs/, and writes everything into
# runs/<run>/second_tests/ so runs/<run>/checkpoints/ keeps the original
# numbers exactly as they were
#
# each (run, round) gets its own eval log, its own overall csv and its own per
# domain csv in there, and a .retested sentinel so the script is safe to
# re-run, anything already done gets skipped
#
# this drives retest_eval.py, which does its own evaluation loop and only
# imports the training scripts for the model and dataset classes, so it does
# not care which commit the repo is sitting on
#
# needs the checkpoints, which were excluded from the rsync down to the laptop
# and only exist on the pod, so run this on a pod that can see /workspace
#
#   bash rerun_tests_nondet_lr1e-5.sh
#
# env overrides
#   REPO_ROOT  where newThesisRepo is           (default /workspace/newThesisRepo)
#   ENV_NAME   conda env                        (default DGOD)
#   SPLIT      test or val                      (default test)
#   ROUNDS     space separated round numbers    (default "0 1 2")

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/newThesisRepo}"
ENV_NAME="${ENV_NAME:-DGOD}"
SPLIT="${SPLIT:-test}"
read -r -a ROUNDS <<< "${ROUNDS:-0 1 2}"

# kind:run_name:weights_prefix
# kind picks which training script retest_eval.py imports and which --exp the
# checkpoint was trained with
RUNS=(
  "baseline:gwhd_baseline_nondet_lr1e-5:baseline_nondet_lr1e-5"
  "dgkarthik:gwhd_dgkarthik_nondet_lr1e-5_reg055:dgkarthik_nondet_lr1e-5_reg055"
)

cd "${REPO_ROOT}"

if [ ! -f retest_eval.py ]; then
  echo "!! retest_eval.py not found in ${REPO_ROOT}, nothing to drive" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# same conda resolution the training scripts use, /workspace first because it
# survives a pod restart
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${ENV_NAME}"

echo "repo:  ${REPO_ROOT} @ $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "split: ${SPLIT}"
echo "rounds: ${ROUNDS[*]}"
echo

for ENTRY in "${RUNS[@]}"; do
  IFS=':' read -r KIND RUN_NAME PREFIX <<< "${ENTRY}"

  RUN_DIR="runs/${RUN_NAME}"
  CKPT_DIR="${RUN_DIR}/checkpoints"
  OUT_DIR="${RUN_DIR}/second_tests"

  if [ ! -d "${CKPT_DIR}" ]; then
    echo "!! Skipping ${RUN_NAME}: no ${CKPT_DIR}"
    continue
  fi

  mkdir -p "${OUT_DIR}"
  {
    echo "retested_at: $(date -Is)"
    echo "git_commit:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "driver:      $(basename "$0")"
    echo "split:       ${SPLIT}"
  } > "${OUT_DIR}/retest_info.txt"

  for ROUND in "${ROUNDS[@]}"; do
    WEIGHTS_FILE="${PREFIX}_round${ROUND}"
    SENTINEL="${OUT_DIR}/${WEIGHTS_FILE}_${SPLIT}.retested"

    if [ -f "${SENTINEL}" ]; then
      echo "== ${RUN_NAME} round ${ROUND}: already retested, skipping"
      continue
    fi

    if [ ! -f "${CKPT_DIR}/${WEIGHTS_FILE}.ckpt" ]; then
      echo "!! ${RUN_NAME} round ${ROUND}: no ${WEIGHTS_FILE}.ckpt, skipping"
      continue
    fi

    echo "== ${RUN_NAME} round ${ROUND}: re-running ${SPLIT} eval =="
    python retest_eval.py \
      --kind "${KIND}" \
      --weights_folder "${CKPT_DIR}" \
      --weights_file "${WEIGHTS_FILE}" \
      --out_dir "${OUT_DIR}" \
      --split "${SPLIT}" \
      2>&1 | tee "${OUT_DIR}/${SPLIT}_round${ROUND}.log"

    touch "${SENTINEL}"
    echo
  done
done

echo "Done. New csvs are in runs/gwhd_*_nondet_lr1e-5*/second_tests/"
echo "Per domain csvs have map_50_pooled (the paper's definition) and"
echo "map_50_image_mean (the old per image average, with the -1 images dropped)."
