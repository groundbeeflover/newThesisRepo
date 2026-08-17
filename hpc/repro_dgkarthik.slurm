#!/bin/bash
#SBATCH --job-name=gwhd_repro_dgkarthik
#SBATCH --output=hpc_logs/stdout-%x-%A_%a.out
#SBATCH --error=hpc_logs/stderr-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=research
#SBATCH --account=research
#SBATCH --array=0-2

# Reproduction of Seemakurthy et al.'s DGKarthik (GRL) number, using
# train_GWHD_dgfrcnn_mattia.py (Mattia Dutto's script, fwd. by Petra Bosilj
# 2026-08-12, patched here to run outside a notebook and to resolve its own
# checkpoint path instead of a hardcoded "GWHD/" -- see git diff / hpc/README.md).
# Settings per the email: BS=8 (hardcoded in the script itself), LR=1e-5,
# alpha_4=0.055, seeds 0/1/2, fully deterministic.
# Target from Mattia's runs: avg test mAP@0.5 ~= 60.3 (see hpc/README.md).
#
# reg_weights (a b c d e): only alpha_4 (d) was specified in the email as
# "the only parameter of note" = 0.055. a/b/c/e are NOT confirmed by Mattia --
# this uses this repo's existing convention (0.5 0.5 0.5 x 0.0001, see
# run_gwhd_dg.sh) with only d swapped to 0.055. If Mattia confirms different
# values for a/b/c/e, update REG_WEIGHTS below.
#
# Array index 0/1/2 -> seed 0/1/2, one job per seed, run concurrently.
#
# Runs on the `research` partition (MaxTime=7-00:00:00 -- scontrol show
# partition research), not `education` (MaxTime=02:00:00, and its
# AllowAccounts=education excludes this account anyway -- see hpc/README.md
# §6). 24h is a generous single-shot budget -- this run took 35-37 epochs in
# earlier GRL attempts (grlbaselineruns/), which fit comfortably inside 24h
# even at a few minutes/epoch.
#
# The training script still checkpoints every epoch and auto-resumes from
# runs/.../checkpoints/last.ckpt if present, and still only writes
# WEIGHTS_FILE.done once val/test eval genuinely finishes -- kept as a safety
# net even though a 24h budget shouldn't need it. Resubmitting the same
# command is always safe: resumes anything unfinished, skips anything
# already .done.
#
# Usage: sbatch hpc/repro_dgkarthik.slurm [lr]
#   lr defaults to 1e-5 (the setting with a known target number). Pass 1e-4
#   to run the other row Mattia mentions exists in the sheet (Sheet3, rows
#   3+5) -- no target number for that one has been shared yet.
# If a task OOMs (L40s have 48GB, so unlikely, but PHYS_BATCH/ACCUM_STEPS are
# here for parity with run_repro_dgkarthik.sh in case you move a seed there):
# sbatch --export=ALL,PHYS_BATCH=4,ACCUM_STEPS=2 hpc/repro_dgkarthik.slurm
set -euo pipefail

LR="${1:-1e-5}"
REG_WEIGHTS=(0.5 0.5 0.5 0.055 0.0001)
SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="gwhd_repro_dgkarthik_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"
WEIGHTS_FILE="dgkarthik_seed${SEED}"

mkdir -p hpc_logs "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

echo "Job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID}) on $(hostname), seed=${SEED}, lr=${LR}, reg_weights=${REG_WEIGHTS[*]}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DGOD

cp train_GWHD_dgfrcnn_mattia.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze_seed${SEED}.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi_seed${SEED}.txt" || true
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ]; then
  echo "Seed ${SEED} already completed (found ${WEIGHTS_FILE}.done). Skipping."
  echo "Delete that file first if you want to force a rerun from scratch."
  exit 0
fi

# NOTE: launched with srun so PyTorch Lightning's auto-detected SLURMEnvironment
# plugin gets the handshake it expects (see hpc/README.md, section 4).
srun python train_GWHD_dgfrcnn_mattia.py \
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
# Separate invocation, separate log: loads the checkpoint just trained and
# writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/.
# Mirrors run_repro_dgkarthik.sh's two-phase pattern -- without this second
# call, no test log or results CSV is ever produced for this seed.
srun python train_GWHD_dgfrcnn_mattia.py \
  --exp dg \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --eval_map \
  --eval_split test \
  2>&1 | tee "${RUN_DIR}/logs/test_seed${SEED}.log"

# Only mark this seed done once both training and eval have succeeded
# (set -euo pipefail above means either failing exits the script first).
touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
