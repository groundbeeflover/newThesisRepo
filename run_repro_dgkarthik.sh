#!/usr/bin/env bash
set -euo pipefail

# RunPod alternative to hpc/repro_dgkarthik.slurm -- runs Mattia Dutto's
# DGKarthik (GRL) reproduction script (train_GWHD_dgfrcnn_mattia.py) for
# seeds 0, 1, 2. Same conda-detection convention as run_gwhd_dg.sh /
# run_gwhd_baseline.sh, and the same runs/gwhd_repro_dgkarthik_lr<LR>/ layout
# hpc/repro_dgkarthik.slurm uses on the university cluster -- so results from
# either source are drop-in comparable, whichever finishes first.
#
# BS=8 (hardcoded inside the script itself), LR=1e-5, alpha_4=0.055 (target
# avg test mAP@0.5 ~= 60.3, see hpc/README.md §8), fully deterministic.
#
# reg_weights (a b c d e): only alpha_4 (d) was specified in Mattia's email as
# "the only parameter of note" = 0.055. a/b/c/e are NOT confirmed -- this uses
# this repo's existing convention (0.5 0.5 0.5 x 0.0001, see run_gwhd_dg.sh)
# with only d swapped to 0.055. Edit REG_WEIGHTS below if Mattia confirms
# different values.
#
# Seeds run SEQUENTIALLY by default -- a single RunPod GPU may not have
# headroom for 3 concurrent BS=8 Faster R-CNN + GRL-head trainings at once.
# See the commented-out block at the bottom for a parallel alternative.
#
# IMPORTANT: unlike sbatch on the cluster, nothing here detaches this from
# your terminal -- it's a plain foreground bash process. Run it inside
# tmux/screen (or nohup) so it survives an SSH disconnect:
#   tmux new -s repro_dgkarthik
#   bash run_repro_dgkarthik.sh
#   # Ctrl-b d to detach; `tmux attach -t repro_dgkarthik` to reattach later
#
# Usage: bash run_repro_dgkarthik.sh [lr]
#   lr defaults to 1e-5 (the setting with a known target number). Pass 1e-4
#   to run the other row Mattia mentions exists in the sheet (Sheet3, rows
#   3+5) -- no target number for that one has been shared yet.

LR="${1:-1e-5}"
ENV_NAME="DGOD"
REG_WEIGHTS=(0.5 0.5 0.5 0.055 0.0001)
SEEDS=(0 1 2)

RUN_NAME="gwhd_repro_dgkarthik_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting DGKarthik reproduction (lr=${LR}, reg_weights=${REG_WEIGHTS[*]}), seeds: ${SEEDS[*]}"

# Prefer persistent conda if installed in /workspace; fall back to ~/miniconda3.
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

  echo "=== dgkarthik seed ${SEED} ==="
  # NOTE: train_GWHD_dgfrcnn_mattia.py checkpoints every epoch and auto-resumes
  # from last.ckpt if this seed was interrupted (e.g. a spot/interruptible pod
  # got reclaimed) -- rerunning this script is always safe. No --num_workers
  # flag here: the script hardcodes num_workers=16 internally, it's not a CLI
  # option (unlike train_GWHD_baseline_clean.py).
  python train_GWHD_dgfrcnn_mattia.py \
    --exp dg \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --reg_weights "${REG_WEIGHTS[@]}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --deterministic \
    2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"
done

echo "Done (or already were). Check ${RUN_DIR}/checkpoints/*.done to confirm which seeds finished."

# --- Optional: run all 3 seeds in parallel instead of sequentially --------
# Only do this if you've confirmed your GPU has enough free VRAM for 3x the
# per-process footprint. num_workers can't be reduced per-process here (see
# note above), so 3 processes x 16 workers each may oversubscribe the pod's
# vCPUs -- watch for degraded throughput if you try this.
#
# for SEED in "${SEEDS[@]}"; do
#   WEIGHTS_FILE="dgkarthik_seed${SEED}"
#   [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ] && continue
#   python train_GWHD_dgfrcnn_mattia.py \
#     --exp dg --weights_folder "${RUN_DIR}/checkpoints" --weights_file "${WEIGHTS_FILE}" \
#     --reg_weights "${REG_WEIGHTS[@]}" --lr "${LR}" --seed "${SEED}" --deterministic \
#     > "${RUN_DIR}/logs/train_seed${SEED}.log" 2>&1 &
# done
# wait
