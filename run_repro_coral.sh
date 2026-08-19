#!/usr/bin/env bash
set -euo pipefail

# runpod deterministic coral reproduction, sibling of run_repro_baseline.sh and
# run_repro_dgkarthik.sh, runs train_GWHD_coralfrcnn.py --exp coral for seeds
# 0, 1 and 2 with --deterministic on, same runs/gwhd_repro_coral_lr<LR>/ layout
# as the other two so everything summarizes the same way
#
# bs=8, lr=1e-5, reg_weights 0.5 0.5 0.5 0.075 0.0001, this repo's usual coral
# convention (see run_gwhd_coral.sh and hpc/train_coral.slurm, the only value
# mattia actually confirmed is dgkarthik's alpha_4=0.055), sampler
# domain_diverse which is the original algorithm, fully deterministic
#
# seeds run one after the other by default rather than at the same time, a single
# runpod gpu probably can't hold 3 concurrent bs=8 coral trainings
#
# unlike sbatch on the cluster nothing here detaches from your terminal, it's
# just a foreground bash process, run it in tmux or screen (or nohup) so an ssh
# drop doesn't kill it:
#   tmux new -s repro_coral
#   bash run_repro_coral.sh
#   #ctrl-b d to detach, tmux attach -t repro_coral to come back
#
# usage: bash run_repro_coral.sh [lr]
#   lr defaults to 1e-5, matching the baseline and dgkarthik deterministic repros
#
# if you get torch.OutOfMemoryError, deterministic coral at bs=8 is tighter again
# than the plain baseline at the same batch size, cudnn.deterministic takes the
# fastest conv algorithms off the table and torch.use_deterministic_algorithms
# forces slower more memory hungry kernels for some ops, same two levers as the
# other two repro scripts, both env vars so you don't have to edit this file:
#
#   PHYS_BATCH / ACCUM_STEPS splits the effective bs=8 into smaller physical
#   steps that get accumulated, so PHYS_BATCH=4 ACCUM_STEPS=2 is still an
#   effective 8 at half the peak memory, keep PHYS_BATCH * ACCUM_STEPS = 8 or the
#   reproduction's effective batch size changes, the default PHYS_BATCH=8
#   ACCUM_STEPS=1 is just the normal behaviour, worth knowing that with
#   --sampler domain_diverse, the default here, this also drops the number of
#   distinct domains guaranteed per physical batch down to PHYS_BATCH, same kind
#   of caveat as batchnorm stats being computed per physical step rather than per
#   effective batch (see --help in train_GWHD_coralfrcnn.py)
#
#   example: PHYS_BATCH=4 ACCUM_STEPS=2 bash run_repro_coral.sh
#
# the PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set below is a free fix
# for memory fragmentation specifically, look for "reserved but unallocated" in
# the oom message, if that number is big this is what you want, if the physical
# batch is just too big it won't save you on its own, use PHYS_BATCH/ACCUM_STEPS
# as well

LR="${1:-1e-5}"
ENV_NAME="DGOD"
REG_WEIGHTS=(0.5 0.5 0.5 0.055 0.0001)
SEEDS=(0 1 2)
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="gwhd_repro_coral_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting CORAL reproduction (lr=${LR}, reg_weights=${REG_WEIGHTS[*]}), seeds: ${SEEDS[*]}"
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

cp train_GWHD_coralfrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp -r dg "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

for SEED in "${SEEDS[@]}"; do
  WEIGHTS_FILE="coral_seed${SEED}"

  if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ]; then
    echo "Seed ${SEED} already completed (found ${WEIGHTS_FILE}.done). Skipping."
    continue
  fi

  echo "=== coral seed ${SEED}: training ==="
  #train_GWHD_coralfrcnn.py checkpoints every epoch and picks back up from
  #<weights_file>-last.ckpt if this seed got interrupted, say a spot pod got
  #reclaimed, so re-running this script is always safe, this call only trains,
  #it writes its own log and its own checkpoint
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --sampler domain_diverse \
    --reg_weights "${REG_WEIGHTS[@]}" \
    --lr "${LR}" \
    --batch_size "${PHYS_BATCH}" \
    --accumulate_grad_batches "${ACCUM_STEPS}" \
    --num_workers 16 \
    --seed "${SEED}" \
    --deterministic \
    2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"

  echo "=== coral seed ${SEED}: test evaluation ==="
  #separate call, separate log, loads the checkpoint that just got trained and
  #writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/
  #train_GWHD_coralfrcnn.py already supported --eval_map, this script just
  #wasn't calling it, which is why coral repro runs never produced a test log or
  #a results csv
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --eval_map \
    --eval_split test \
    2>&1 | tee "${RUN_DIR}/logs/test_seed${SEED}.log"

  #only mark the seed done once training and eval have both worked, set -euo
  #pipefail up top means either one failing kills the script before we get here
  #train_GWHD_coralfrcnn.py doesn't write its own WEIGHTS_FILE.done sentinel the
  #way the baseline and mattia scripts do, so write one here and the
  #skip-if-done check above behaves the same
  touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
done

echo "Done (or already were). Check ${RUN_DIR}/checkpoints/*.done to confirm which seeds finished."
echo "Per-seed results: ${RUN_DIR}/checkpoints/coral_seed*_test_map.csv"

# --- optional, run all 3 seeds at once instead of one after the other ------
# only bother if you've actually checked the gpu has 3x the per process vram
# free, and drop --num_workers per process (5 or so) so three of them don't
# fight over the pod's vcpus
#
# for SEED in "${SEEDS[@]}"; do
#   WEIGHTS_FILE="coral_seed${SEED}"
#   [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ] && continue
#   python train_GWHD_coralfrcnn.py \
#     --exp coral --weights_folder "${RUN_DIR}/checkpoints" --weights_file "${WEIGHTS_FILE}" \
#     --sampler domain_diverse --reg_weights "${REG_WEIGHTS[@]}" \
#     --lr "${LR}" --batch_size "${PHYS_BATCH}" --accumulate_grad_batches "${ACCUM_STEPS}" \
#     --num_workers 5 --seed "${SEED}" --deterministic \
#     > "${RUN_DIR}/logs/train_seed${SEED}.log" 2>&1 && touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" &
# done
# wait
