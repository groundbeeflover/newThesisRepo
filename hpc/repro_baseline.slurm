#!/bin/bash
#SBATCH --job-name=gwhd_repro_baseline
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

# Reproduction of Seemakurthy et al.'s baseline number, using the clean/stock
# torchvision Faster R-CNN in train_GWHD_baseline_clean.py (NOT the repo's
# custom fasterrcnn.py/non_dg path) -- per Mattia Dutto's email, fwd. by
# Petra Bosilj, 2026-08-12: BS=8, LR=1e-5, seeds 0/1/2, fully deterministic.
# Target from Mattia's runs: avg test mAP@0.5 ~= 54.7 (see hpc/README.md).
#
# Array index 0/1/2 -> seed 0/1/2, one job per seed, run concurrently.
#
# Runs on the `research` partition (MaxTime=7-00:00:00 -- scontrol show
# partition research), not `education` (MaxTime=02:00:00, and its
# AllowAccounts=education excludes this account anyway -- see hpc/README.md
# §6). 24h is a generous single-shot budget for 100 epochs/patience-10 early
# stopping; the earlier grlbaselineruns baseline converged in 13-23 epochs.
#
# The training script still checkpoints every epoch and auto-resumes from
# runs/.../checkpoints/last.ckpt if present, and still only writes
# WEIGHTS_FILE.done once val/test eval genuinely finishes -- kept as a safety
# net (job gets killed, node reboots, you hit a real per-job time limit later)
# even though a 24h budget shouldn't need it. Resubmitting the same command is
# always safe: resumes anything unfinished, skips anything already .done.
#
# Usage: sbatch hpc/repro_baseline.slurm [lr]
#   lr defaults to 1e-5 (the setting with a known target number). Pass 1e-4
#   to run the other row Mattia mentions exists in the sheet (Sheet3, rows
#   3+5) -- no target number for that one has been shared yet.
# If a task OOMs (L40s have 48GB, so unlikely, but PHYS_BATCH/ACCUM_STEPS are
# here for parity with run_repro_baseline.sh in case you move a seed there):
# sbatch --export=ALL,PHYS_BATCH=4,ACCUM_STEPS=2 hpc/repro_baseline.slurm
set -euo pipefail

LR="${1:-1e-5}"
SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="gwhd_repro_baseline_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"
WEIGHTS_FILE="baseline_seed${SEED}"

mkdir -p hpc_logs "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

echo "Job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID}) on $(hostname), seed=${SEED}, lr=${LR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DGOD

cp train_GWHD_baseline_clean.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
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
srun python train_GWHD_baseline_clean.py \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --lr "${LR}" \
  --batch_size "${PHYS_BATCH}" \
  --accumulate_grad_batches "${ACCUM_STEPS}" \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --seed "${SEED}" \
  --deterministic \
  2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"
