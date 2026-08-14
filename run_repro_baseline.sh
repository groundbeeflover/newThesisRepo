#!/usr/bin/env bash
set -euo pipefail

# RunPod alternative to hpc/repro_baseline.slurm -- runs the clean/stock
# torchvision Faster R-CNN baseline reproduction (train_GWHD_baseline_clean.py)
# for seeds 0, 1, 2. Same conda-detection convention as run_gwhd_dg.sh /
# run_gwhd_baseline.sh, and the same runs/gwhd_repro_baseline_lr<LR>/ layout
# hpc/repro_baseline.slurm uses on the university cluster -- so results from
# either source are drop-in comparable, whichever finishes first.
#
# BS=8, LR=1e-5 (Mattia's setting; target avg test mAP@0.5 ~= 54.7, see
# hpc/README.md §8), fully deterministic.
#
# Seeds run SEQUENTIALLY by default, not concurrently -- a single RunPod GPU
# may not have headroom for 3 concurrent BS=8 Faster R-CNN trainings at once.
# If your pod's GPU is large (>=40GB) and you want to try parallel instead,
# see the commented-out block at the bottom.
#
# IMPORTANT: unlike sbatch on the cluster, nothing here detaches this from
# your terminal -- it's a plain foreground bash process. Run it inside
# tmux/screen (or nohup) so it survives an SSH disconnect:
#   tmux new -s repro_baseline
#   bash run_repro_baseline.sh
#   # Ctrl-b d to detach; `tmux attach -t repro_baseline` to reattach later
#
# Usage: bash run_repro_baseline.sh [lr]
#   lr defaults to 1e-5 (the setting with a known target number). Pass 1e-4
#   to run the other row Mattia mentions exists in the sheet (Sheet3, rows
#   3+5) -- no target number for that one has been shared yet.

LR="${1:-1e-5}"
ENV_NAME="DGOD"
SEEDS=(0 1 2)

RUN_NAME="gwhd_repro_baseline_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting clean-baseline reproduction (lr=${LR}), seeds: ${SEEDS[*]}"

# Prefer persistent conda if installed in /workspace; fall back to ~/miniconda3.
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

  echo "=== baseline seed ${SEED} ==="
  # NOTE: train_GWHD_baseline_clean.py checkpoints every epoch and auto-resumes
  # from last.ckpt if this seed was interrupted (e.g. a spot/interruptible pod
  # got reclaimed) -- rerunning this script is always safe.
  python train_GWHD_baseline_clean.py \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --lr "${LR}" \
    --batch_size 8 \
    --num_workers 16 \
    --seed "${SEED}" \
    --deterministic \
    2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"
done

echo "Done (or already were). Check ${RUN_DIR}/checkpoints/*.done to confirm which seeds finished."

# --- Optional: run all 3 seeds in parallel instead of sequentially --------
# Only do this if you've confirmed your GPU has enough free VRAM for 3x the
# per-process footprint, and reduce --num_workers per process (e.g. to 5) so
# you don't oversubscribe the pod's vCPUs across all three at once.
#
# for SEED in "${SEEDS[@]}"; do
#   WEIGHTS_FILE="baseline_seed${SEED}"
#   [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ] && continue
#   python train_GWHD_baseline_clean.py \
#     --weights_folder "${RUN_DIR}/checkpoints" --weights_file "${WEIGHTS_FILE}" \
#     --lr "${LR}" --batch_size 8 --num_workers 5 --seed "${SEED}" --deterministic \
#     > "${RUN_DIR}/logs/train_seed${SEED}.log" 2>&1 &
# done
# wait
