#!/bin/bash
#SBATCH --job-name=gwhd_repro_dgkarthik
#SBATCH --output=hpc_logs/stdout-%x-%A_%a.out
#SBATCH --error=hpc_logs/stderr-%x-%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=education
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
# Usage: sbatch hpc/repro_dgkarthik.slurm [lr]
#   lr defaults to 1e-5 (the setting with a known target number). Pass 1e-4
#   to run the other row Mattia mentions exists in the sheet (Sheet3, rows
#   3+5) -- no target number for that one has been shared yet.
set -euo pipefail

LR="${1:-1e-5}"
REG_WEIGHTS=(0.5 0.5 0.5 0.055 0.0001)
SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

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

if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt" ]; then
  echo "ERROR: Checkpoint already exists: ${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt"
  echo "Delete it first if you want to rerun this seed."
  exit 1
fi

# NOTE: launched with srun so PyTorch Lightning's auto-detected SLURMEnvironment
# plugin gets the handshake it expects (see hpc/README.md, section 4).
srun python train_GWHD_dgfrcnn_mattia.py \
  --exp dg \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --reg_weights "${REG_WEIGHTS[@]}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --deterministic \
  2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"
