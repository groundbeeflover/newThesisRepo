#!/usr/bin/env bash
set -euo pipefail

# runpod version of hpc/repro_dgkarthik.slurm, runs mattia dutto's dgkarthik
# (grl) reproduction script (train_GWHD_dgfrcnn_mattia.py) for seeds 0, 1 and 2
# same runs/gwhd_repro_dgkarthik_lr<LR>/ layout the slurm script uses on the uni
# cluster, so whichever one finishes first the results drop straight into the
# same comparison
#
# bs=8, hardcoded inside the script itself, lr=1e-5, alpha_4=0.055 (target avg
# test map@0.5 around 60.3, see hpc/README.md section 8), fully deterministic
#
# on reg_weights (a b c d e): mattia's email only ever specified alpha_4 (d) as
# "the only parameter of note" = 0.055, a/b/c/e are not confirmed, so this uses
# this repo's existing convention, 0.5 0.5 0.5 x 0.0001 (see run_gwhd_dg.sh),
# with only d swapped over, change REG_WEIGHTS below if mattia confirms anything
# different
#
# seeds run one after the other by default, a single runpod gpu probably can't
# hold 3 concurrent bs=8 faster r-cnn + grl head trainings, there's a commented
# out parallel version at the bottom
#
# unlike sbatch on the cluster nothing here detaches from your terminal, it's
# just a foreground bash process, run it in tmux or screen (or nohup) so an ssh
# drop doesn't kill it:
#   tmux new -s repro_dgkarthik
#   bash run_repro_dgkarthik.sh
#   #ctrl-b d to detach, tmux attach -t repro_dgkarthik to come back
#
# usage: bash run_repro_dgkarthik.sh [lr]
#   lr defaults to 1e-5, the one with a known target number, pass 1e-4 to run the
#   other row mattia says is in the sheet (sheet3, rows 3+5), nobody's shared a
#   target for that one yet
#
# if you get torch.OutOfMemoryError, and this run is hungrier than the baseline
# since it's the same bs=8 detector with the da heads bolted on, so if the
# baseline oom'd on your card expect this to as well, same two levers as
# run_repro_baseline.sh and both are env vars:
#
#   PHYS_BATCH / ACCUM_STEPS splits the effective bs=8 into smaller physical
#   steps that get accumulated, keep PHYS_BATCH * ACCUM_STEPS = 8 or the
#   reproduction's effective batch size changes
#   example: PHYS_BATCH=4 ACCUM_STEPS=2 bash run_repro_dgkarthik.sh
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True below is a free fix for
# fragmentation specifically, look for a big "reserved but unallocated" in the
# oom message, use PHYS_BATCH/ACCUM_STEPS as well if the physical batch is just
# too big for the card

LR="${1:-1e-5}"
ENV_NAME="DGOD"
REG_WEIGHTS=(0.5 0.5 0.5 0.055 0.0001)
SEEDS=(0 1 2)
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="gwhd_repro_dgkarthik_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting DGKarthik reproduction (lr=${LR}, reg_weights=${REG_WEIGHTS[*]}), seeds: ${SEEDS[*]}"
echo "Physical batch=${PHYS_BATCH}, accumulate_grad_batches=${ACCUM_STEPS} (effective batch=$((PHYS_BATCH * ACCUM_STEPS)))"

#use the conda in /workspace if it's there, it survives a pod restart,
#otherwise fall back to the one in home
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${ENV_NAME}"

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

cp train_GWHD_dgfrcnn_mattia.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

for SEED in "${SEEDS[@]}"; do
  WEIGHTS_FILE="dgkarthik_seed${SEED}"

  if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ]; then
    echo "Seed ${SEED} already completed (found ${WEIGHTS_FILE}.done). Skipping."
    continue
  fi

  echo "=== dgkarthik seed ${SEED}: training ==="
  #train_GWHD_dgfrcnn_mattia.py checkpoints every epoch and picks back up from
  #<weights_file>-last.ckpt if this seed got interrupted, say a spot pod got
  #reclaimed, so re-running this script is always safe, no --num_workers here,
  #that script hardcodes num_workers=16 internally and doesn't expose it on the
  #cli the way train_GWHD_baseline_clean.py does, this call only trains, it
  #writes its own log and its own checkpoint
  python train_GWHD_dgfrcnn_mattia.py \
    --exp dg \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --reg_weights "${REG_WEIGHTS[@]}" \
    --lr "${LR}" \
    --batch_size "${PHYS_BATCH}" \
    --accumulate_grad_batches "${ACCUM_STEPS}" \
    --seed "${SEED}" \
    --deterministic \
    2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"

  echo "=== dgkarthik seed ${SEED}: test evaluation ==="
  #separate call, separate log, loads the checkpoint that just got trained and
  #writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/
  python train_GWHD_dgfrcnn_mattia.py \
    --exp dg \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --eval_map \
    --eval_split test \
    2>&1 | tee "${RUN_DIR}/logs/test_seed${SEED}.log"

  #only mark the seed done once training and eval have both worked, set -euo
  #pipefail up top means either one failing kills the script before we get here
  touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
done

echo "Done (or already were). Check ${RUN_DIR}/checkpoints/*.done to confirm which seeds finished."
echo "Per-seed results: ${RUN_DIR}/checkpoints/dgkarthik_seed*_test_map.csv"

# --- optional, run all 3 seeds at once instead of one after the other ------
# only bother if you've actually checked the gpu has 3x the per process vram
# free, num_workers can't be turned down per process here (see the note above),
# so 3 processes at 16 workers each might swamp the pod's vcpus, keep an eye out
# for throughput dropping if you try it
#
# for SEED in "${SEEDS[@]}"; do
#   WEIGHTS_FILE="dgkarthik_seed${SEED}"
#   [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ] && continue
#   python train_GWHD_dgfrcnn_mattia.py \
#     --exp dg --weights_folder "${RUN_DIR}/checkpoints" --weights_file "${WEIGHTS_FILE}" \
#     --reg_weights "${REG_WEIGHTS[@]}" --lr "${LR}" --batch_size "${PHYS_BATCH}" \
#     --accumulate_grad_batches "${ACCUM_STEPS}" --seed "${SEED}" --deterministic \
#     > "${RUN_DIR}/logs/train_seed${SEED}.log" 2>&1 &
# done
# wait
