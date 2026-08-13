#!/bin/bash
#SBATCH --job-name=gwhd_repro_baseline
#SBATCH --output=hpc_logs/stdout-%x-%A_%a.out
#SBATCH --error=hpc_logs/stderr-%x-%A_%a.err
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=education
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
# The `education` partition caps walltime at 2h (scontrol show partition
# education -> MaxTime=02:00:00), well short of what 100 epochs with
# patience-10 early stopping needs. --time is set to 01:55:00 (5 min buffer
# for env setup + checkpoint writing) and the training script auto-resumes
# from runs/.../checkpoints/last.ckpt if present -- so just RESUBMIT THE SAME
# COMMAND AGAIN after each 2h chunk times out, as many times as it takes:
#
#   sbatch hpc/repro_baseline.slurm
#
# Re-running is safe: it either resumes an in-progress seed or, if a seed's
# WEIGHTS_FILE.done sentinel already exists (see below), skips it as already
# complete instead of erroring.
#
# Usage: sbatch hpc/repro_baseline.slurm [lr]
#   lr defaults to 1e-5 (the setting with a known target number). Pass 1e-4
#   to run the other row Mattia mentions exists in the sheet (Sheet3, rows
#   3+5) -- no target number for that one has been shared yet.
set -euo pipefail

LR="${1:-1e-5}"
SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

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
  --batch_size 8 \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --seed "${SEED}" \
  --deterministic \
  2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"
