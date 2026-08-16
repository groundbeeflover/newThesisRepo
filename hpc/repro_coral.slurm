#!/bin/bash
#SBATCH --job-name=gwhd_repro_coral
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

# Cluster counterpart of run_repro_coral.sh -- deterministic CORAL run using
# train_GWHD_coralfrcnn.py --exp coral: BS=8, LR=1e-5, reg_weights
# 0.5 0.5 0.5 0.075 0.0001 (this repo's existing CORAL convention -- see
# run_gwhd_coral.sh / hpc/train_coral.slurm), sampler domain_diverse, seeds
# 0/1/2, fully deterministic. Unlike hpc/repro_baseline.slurm and
# hpc/repro_dgkarthik.slurm, there is no external paper target number for
# CORAL to reproduce -- this exists for parity with those two so all three
# deterministic BS=8 runs (baseline / DGKarthik-GRL / CORAL) are comparable
# to each other under identical seeding/determinism/VRAM-lever handling.
#
# Array index 0/1/2 -> seed 0/1/2, one job per seed, run concurrently.
#
# Runs on the `research` partition (MaxTime=7-00:00:00), not `education`
# (MaxTime=02:00:00) -- see hpc/README.md §6. 24h is a generous single-shot
# budget for 100 epochs/patience-10 early stopping.
#
# The training script checkpoints every epoch and auto-resumes from
# runs/.../checkpoints/last.ckpt if present. Resubmitting the same command is
# always safe: resumes anything unfinished, skips anything already .done
# (train_GWHD_coralfrcnn.py doesn't write that sentinel itself -- this script
# writes it after a successful run, same as run_repro_coral.sh).
#
# Usage: sbatch hpc/repro_coral.slurm [lr]
#   lr defaults to 1e-5, matching hpc/repro_baseline.slurm / repro_dgkarthik.slurm.
# If a task OOMs (L40s have 48GB, so unlikely, but PHYS_BATCH/ACCUM_STEPS are
# here for parity with run_repro_coral.sh in case you move a seed there):
# sbatch --export=ALL,PHYS_BATCH=4,ACCUM_STEPS=2 hpc/repro_coral.slurm
set -euo pipefail

LR="${1:-1e-5}"
REG_WEIGHTS=(0.5 0.5 0.5 0.075 0.0001)
SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="gwhd_repro_coral_lr${LR}"
RUN_DIR="runs/${RUN_NAME}"
WEIGHTS_FILE="coral_seed${SEED}"

mkdir -p hpc_logs "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

echo "Job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID}) on $(hostname), seed=${SEED}, lr=${LR}, reg_weights=${REG_WEIGHTS[*]}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DGOD

cp train_GWHD_coralfrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp -r dg "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
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
srun python train_GWHD_coralfrcnn.py \
  --exp coral \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --sampler domain_diverse \
  --reg_weights "${REG_WEIGHTS[@]}" \
  --lr "${LR}" \
  --batch_size "${PHYS_BATCH}" \
  --accumulate_grad_batches "${ACCUM_STEPS}" \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --seed "${SEED}" \
  --deterministic \
  2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"

echo "=== coral seed ${SEED}: test evaluation ==="
# Separate invocation, separate log: loads the checkpoint just trained and
# writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/.
# Mirrors run_repro_coral.sh's two-phase pattern -- without this second call,
# no test log or results CSV is ever produced for this seed.
srun python train_GWHD_coralfrcnn.py \
  --exp coral \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --eval_map \
  --eval_split test \
  2>&1 | tee "${RUN_DIR}/logs/test_seed${SEED}.log"

# Only mark this seed done once both training and eval have succeeded
# (set -euo pipefail above means either failing exits the script first).
touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
