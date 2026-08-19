#!/usr/bin/env bash
set -euo pipefail

# runpod version of hpc/repro_baseline.slurm, runs the clean stock torchvision
# faster r-cnn baseline reproduction (train_GWHD_baseline_clean.py) for seeds
# 0, 1 and 2, same runs/gwhd_repro_baseline_lr<LR>/ layout the slurm script uses
# on the uni cluster, so whichever one finishes first the results drop straight
# into the same comparison
#
# bs=8, lr=1e-5 (mattia's setting, target avg test map@0.5 around 54.7, see
# hpc/README.md section 8), fully deterministic
#
# seeds run one after the other by default rather than at the same time, a single
# runpod gpu probably can't hold 3 concurrent bs=8 faster r-cnn trainings, if
# your pod's gpu is big (40gb+) and you want to try it in parallel, there's a
# commented out block at the bottom
#
# unlike sbatch on the cluster nothing here detaches from your terminal, it's
# just a foreground bash process, run it in tmux or screen (or nohup) so an ssh
# drop doesn't kill it:
#   tmux new -s repro_baseline
#   bash run_repro_baseline.sh
#   #ctrl-b d to detach, tmux attach -t repro_baseline to come back
#
# usage: bash run_repro_baseline.sh [lr]
#   lr defaults to 1e-5, the one with a known target number, pass 1e-4 to run the
#   other row mattia says is in the sheet (sheet3, rows 3+5), nobody's shared a
#   target for that one yet
#
# if you get torch.OutOfMemoryError, which happens a lot on a 24gb card at bs=8
# with 1024x1024 images and is a genuinely tight fit rather than a bug, there are
# two separate levers and both are env vars so you don't have to edit this file:
#
#   PHYS_BATCH / ACCUM_STEPS splits the effective bs=8 into smaller physical
#   steps that get accumulated, so PHYS_BATCH=4 ACCUM_STEPS=2 is still an
#   effective 8 at half the peak memory, keep PHYS_BATCH * ACCUM_STEPS = 8 or the
#   reproduction's effective batch size changes, the default PHYS_BATCH=8
#   ACCUM_STEPS=1 is just the normal behaviour
#
#   example: PHYS_BATCH=4 ACCUM_STEPS=2 bash run_repro_baseline.sh
#
# the PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set below is a free fix
# for memory fragmentation specifically, look for "reserved but unallocated" in
# the oom message, if that number is big this is what you want, if the physical
# batch is just too big it won't save you on its own, use PHYS_BATCH/ACCUM_STEPS
# as well

LR="${1:-1e-5}"
ENV_NAME="DGOD"
SEEDS=(0 1 2)
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="gwhd_repro_baseline_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting clean-baseline reproduction (lr=${LR}), seeds: ${SEEDS[*]}"
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

cp train_GWHD_baseline_clean.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

for SEED in "${SEEDS[@]}"; do
  WEIGHTS_FILE="baseline_seed${SEED}"

  if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ]; then
    echo "Seed ${SEED} already completed (found ${WEIGHTS_FILE}.done). Skipping."
    continue
  fi

  echo "=== baseline seed ${SEED}: training ==="
  #train_GWHD_baseline_clean.py checkpoints every epoch and picks back up from
  #<weights_file>-last.ckpt if this seed got interrupted, say a spot pod got
  #reclaimed, so re-running this script is always safe, this call only trains,
  #it writes this seed's train_seed*.log and ${WEIGHTS_FILE}.ckpt and never
  #touches the test data
  python train_GWHD_baseline_clean.py \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --lr "${LR}" \
    --batch_size "${PHYS_BATCH}" \
    --accumulate_grad_batches "${ACCUM_STEPS}" \
    --num_workers 16 \
    --seed "${SEED}" \
    --deterministic \
    2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"

  echo "=== baseline seed ${SEED}: test evaluation ==="
  #separate call, separate log, loads the checkpoint that just got trained and
  #writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/
  python train_GWHD_baseline_clean.py \
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
echo "Per-seed results: ${RUN_DIR}/checkpoints/baseline_seed*_test_map.csv"

# --- optional, run all 3 seeds at once instead of one after the other ------
# only bother if you've actually checked the gpu has 3x the per process vram
# free, and drop --num_workers per process (5 or so) so three of them don't
# fight over the pod's vcpus
#
# for SEED in "${SEEDS[@]}"; do
#   WEIGHTS_FILE="baseline_seed${SEED}"
#   [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ] && continue
#   python train_GWHD_baseline_clean.py \
#     --weights_folder "${RUN_DIR}/checkpoints" --weights_file "${WEIGHTS_FILE}" \
#     --lr "${LR}" --batch_size "${PHYS_BATCH}" --accumulate_grad_batches "${ACCUM_STEPS}" \
#     --num_workers 5 --seed "${SEED}" --deterministic \
#     > "${RUN_DIR}/logs/train_seed${SEED}.log" 2>&1 &
# done
# wait
