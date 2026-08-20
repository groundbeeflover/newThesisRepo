#!/usr/bin/env bash
# runpod. same idea as rerun_tests_nondet_lr1e-5.sh but for the coral runs, and
# it discovers them instead of hardcoding a list, because which coral runs have
# usable checkpoints changes while training is still going
#
# a (run, round) is picked up when both of these are true
#   runs/gwhd_coral_*/checkpoints/<weights_file>.ckpt exists
#   runs/gwhd_coral_*/checkpoints/<weights_file>.done exists
# the .done is what the training script drops after a round finishes, so a
# round that is still training is skipped rather than evaluated half trained.
# last.ckpt is ignored, it is the resume checkpoint and not the best val_acc one
#
# results go to runs/<run>/second_tests/, the original checkpoints/ csvs are
# never touched, and a .retested sentinel makes the script re-runnable, so it
# is fine to run it now for the finished runs and again later once the rest
# land
#
#   bash rerun_tests_coral.sh
#
# env overrides
#   REPO_ROOT   where newThesisRepo is         (default /workspace/newThesisRepo)
#   ENV_NAME    conda env                      (default DGOD)
#   SPLIT       test or val                    (default test)
#   RUN_GLOB    which run dirs to consider     (default runs/gwhd_coral_*)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/newThesisRepo}"
ENV_NAME="${ENV_NAME:-DGOD}"
SPLIT="${SPLIT:-test}"
RUN_GLOB="${RUN_GLOB:-runs/gwhd_coral_*}"

cd "${REPO_ROOT}"

if [ ! -f retest_eval.py ]; then
  echo "!! retest_eval.py not found in ${REPO_ROOT}, nothing to drive" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${ENV_NAME}"

echo "repo:  ${REPO_ROOT} @ $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "split: ${SPLIT}"
echo "glob:  ${RUN_GLOB}"
echo

FOUND=0
SKIPPED_UNFINISHED=0

for RUN_DIR in ${RUN_GLOB}; do
  [ -d "${RUN_DIR}/checkpoints" ] || continue

  RUN_NAME="$(basename "${RUN_DIR}")"
  CKPT_DIR="${RUN_DIR}/checkpoints"
  OUT_DIR="${RUN_DIR}/second_tests"

  # every checkpoint in this run except the resume one
  shopt -s nullglob
  CKPTS=("${CKPT_DIR}"/*.ckpt)
  shopt -u nullglob
  [ ${#CKPTS[@]} -gt 0 ] || continue

  HEADER_DONE=0

  for CKPT in "${CKPTS[@]}"; do
    WEIGHTS_FILE="$(basename "${CKPT}" .ckpt)"
    [ "${WEIGHTS_FILE}" = "last" ] && continue

    # a round that has not finished training has no .done, evaluating its
    # checkpoint would silently mix a partly trained model into the results
    if [ ! -f "${CKPT_DIR}/${WEIGHTS_FILE}.done" ]; then
      echo "-- ${RUN_NAME}/${WEIGHTS_FILE}: no .done, still training, skipping"
      SKIPPED_UNFINISHED=$((SKIPPED_UNFINISHED + 1))
      continue
    fi

    SENTINEL="${OUT_DIR}/${WEIGHTS_FILE}_${SPLIT}.retested"
    if [ -f "${SENTINEL}" ]; then
      echo "== ${RUN_NAME}/${WEIGHTS_FILE}: already retested, skipping"
      continue
    fi

    mkdir -p "${OUT_DIR}"
    if [ "${HEADER_DONE}" -eq 0 ]; then
      {
        echo "retested_at: $(date -Is)"
        echo "git_commit:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "driver:      $(basename "$0")"
        echo "split:       ${SPLIT}"
      } > "${OUT_DIR}/retest_info.txt"
      HEADER_DONE=1
    fi

    echo "== ${RUN_NAME}/${WEIGHTS_FILE}: re-running ${SPLIT} eval =="
    python retest_eval.py \
      --kind coral \
      --weights_folder "${CKPT_DIR}" \
      --weights_file "${WEIGHTS_FILE}" \
      --out_dir "${OUT_DIR}" \
      --split "${SPLIT}" \
      2>&1 | tee "${OUT_DIR}/${SPLIT}_${WEIGHTS_FILE}.log"

    touch "${SENTINEL}"
    FOUND=$((FOUND + 1))
    echo
  done
done

echo "Done. Re-evaluated ${FOUND} checkpoint(s)."
if [ "${SKIPPED_UNFINISHED}" -gt 0 ]; then
  echo "${SKIPPED_UNFINISHED} checkpoint(s) skipped for having no .done yet."
  echo "Re-run this script once those rounds finish, the sentinels stop it"
  echo "from redoing the ones it already did."
fi
