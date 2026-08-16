#!/usr/bin/env bash
set -euo pipefail

# RunPod deterministic CORAL reproduction run, sibling to run_repro_baseline.sh /
# run_repro_dgkarthik.sh -- runs train_GWHD_coralfrcnn.py --exp coral for
# seeds 0, 1, 2 with --deterministic set. Same conda-detection convention and
# the same runs/gwhd_repro_coral_lr<LR>/ layout as the other two, so results
# are drop-in comparable / summarizable the same way.
#
# BS=8, LR=1e-5, reg_weights 0.5 0.5 0.5 0.075 0.0001 (this repo's existing
# CORAL convention -- see run_gwhd_coral.sh / hpc/train_coral.slurm; only
# dgkarthik's alpha_4=0.055 is a value Mattia actually confirmed), sampler
# domain_diverse (the original algorithm), fully deterministic.
#
# Seeds run SEQUENTIALLY by default, not concurrently -- a single RunPod GPU
# may not have headroom for 3 concurrent BS=8 CORAL trainings at once.
#
# IMPORTANT: unlike sbatch on the cluster, nothing here detaches this from
# your terminal -- it's a plain foreground bash process. Run it inside
# tmux/screen (or nohup) so it survives an SSH disconnect:
#   tmux new -s repro_coral
#   bash run_repro_coral.sh
#   # Ctrl-b d to detach; `tmux attach -t repro_coral` to reattach later
#
# Usage: bash run_repro_coral.sh [lr]
#   lr defaults to 1e-5, matching the baseline/dgkarthik deterministic repros.
#
# If you hit `torch.OutOfMemoryError`: deterministic CORAL at BS=8 is even
# tighter than the plain baseline at the same batch size -- cudnn.deterministic
# disables the fastest conv algorithms, and torch.use_deterministic_algorithms
# forces slower, more memory-hungry deterministic kernels for some ops. Same
# two independent levers as run_repro_baseline.sh / run_repro_dgkarthik.sh,
# both settable via env vars without editing this file:
#
#   PHYS_BATCH / ACCUM_STEPS -- split the effective BS=8 into smaller physical
#   steps accumulated together (e.g. PHYS_BATCH=4 ACCUM_STEPS=2 = effective 8,
#   half the peak memory). Keep PHYS_BATCH * ACCUM_STEPS = 8 to preserve the
#   reproduction's effective batch size. Default PHYS_BATCH=8 ACCUM_STEPS=1 is
#   unchanged behavior. Note: for --sampler domain_diverse (the default here),
#   this also shrinks the number of distinct domains guaranteed per physical
#   batch to PHYS_BATCH -- same caveat as BatchNorm statistics being computed
#   per physical step, not per effective batch (see --help in
#   train_GWHD_coralfrcnn.py).
#
#   Example: PHYS_BATCH=4 ACCUM_STEPS=2 bash run_repro_coral.sh
#
# The PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set below is a free,
# semantics-preserving fix for memory *fragmentation* specifically (look for
# "reserved but unallocated" in the OOM error -- if that number is large, this
# is the fix; if it's the PHYS_BATCH that's actually too big, it won't help on
# its own, use PHYS_BATCH/ACCUM_STEPS above too).

LR="${1:-1e-5}"
ENV_NAME="DGOD"
REG_WEIGHTS=(0.5 0.5 0.5 0.075 0.0001)
SEEDS=(0 1 2)
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="gwhd_repro_coral_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting CORAL reproduction (lr=${LR}, reg_weights=${REG_WEIGHTS[*]}), seeds: ${SEEDS[*]}"
echo "Physical batch=${PHYS_BATCH}, accumulate_grad_batches=${ACCUM_STEPS} (effective batch=$((PHYS_BATCH * ACCUM_STEPS)))"

# Prefer persistent conda if installed in /workspace; fall back to ~/miniconda3.
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
  # NOTE: train_GWHD_coralfrcnn.py checkpoints every epoch and auto-resumes
  # from <weights_file>-last.ckpt if this seed was interrupted (e.g. a
  # spot/interruptible pod got reclaimed) -- rerunning this script is always
  # safe. Training-only: writes its own log and its own checkpoint.
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
  # Separate invocation, separate log: loads the checkpoint just trained and
  # writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/.
  # (train_GWHD_coralfrcnn.py already had --eval_map support; this script
  # just wasn't calling it before, so no test log or results CSV was ever
  # produced for CORAL repro runs.)
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --eval_map \
    --eval_split test \
    2>&1 | tee "${RUN_DIR}/logs/test_seed${SEED}.log"

  # Only mark this seed done once both training and eval have succeeded
  # (set -euo pipefail above means either failing exits the script first).
  # train_GWHD_coralfrcnn.py doesn't write a WEIGHTS_FILE.done sentinel itself
  # (unlike train_GWHD_baseline_clean.py / train_GWHD_dgfrcnn_mattia.py) --
  # write one here so the skip-if-done check above works the same way.
  touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
done

echo "Done (or already were). Check ${RUN_DIR}/checkpoints/*.done to confirm which seeds finished."
echo "Per-seed results: ${RUN_DIR}/checkpoints/coral_seed*_test_map.csv"

# --- Optional: run all 3 seeds in parallel instead of sequentially --------
# Only do this if you've confirmed your GPU has enough free VRAM for 3x the
# per-process footprint, and reduce --num_workers per process (e.g. to 5) so
# you don't oversubscribe the pod's vCPUs across all three at once.
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
