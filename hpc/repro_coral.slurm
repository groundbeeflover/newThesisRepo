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

# cluster version of run_repro_coral.sh, deterministic coral run off
# train_GWHD_coralfrcnn.py --exp coral, bs=8, lr=1e-5, reg_weights
# 0.5 0.5 0.5 0.075 0.0001 which is this repo's usual coral convention (see
# run_gwhd_coral.sh and hpc/train_coral.slurm), sampler domain_diverse, seeds
# 0/1/2, fully deterministic, unlike hpc/repro_baseline.slurm and
# hpc/repro_dgkarthik.slurm there's no external paper number for coral to
# reproduce, this exists so all three deterministic bs=8 runs, baseline,
# dgkarthik-grl and coral, are comparable to each other under identical seeding,
# determinism and vram handling
#
# array index 0/1/2 maps to seed 0/1/2, one job each, all running at once
#
# runs on the research partition (MaxTime=7-00:00:00) rather than education
# (MaxTime=02:00:00), see hpc/README.md section 6, 24h is a generous one shot
# budget for 100 epochs with patience 10 early stopping
#
# the training script checkpoints every epoch and picks back up from
# runs/.../checkpoints/last.ckpt if it's there, resubmitting the same command is
# always safe, it resumes whatever's unfinished and skips whatever's already
# .done, though train_GWHD_coralfrcnn.py doesn't write that sentinel itself so
# this script writes it after a successful run, same as run_repro_coral.sh
#
# usage: sbatch hpc/repro_coral.slurm [lr]
#   lr defaults to 1e-5, matching repro_baseline.slurm and repro_dgkarthik.slurm
# if a task ooms, and l40s have 48gb so it shouldn't, PHYS_BATCH/ACCUM_STEPS are
# here to match run_repro_coral.sh in case you move a seed over there:
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

#launched with srun so lightning's auto-detected slurm environment plugin
#actually gets the handshake it wants (see hpc/README.md, section 4)
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
#separate call, separate log, loads the checkpoint that just got trained and
#writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/, the
#same two phase thing run_repro_coral.sh does, without this second call this
#seed never produces a test log or a results csv at all
srun python train_GWHD_coralfrcnn.py \
  --exp coral \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --eval_map \
  --eval_split test \
  2>&1 | tee "${RUN_DIR}/logs/test_seed${SEED}.log"

#only mark the seed done once training and eval have both worked, set -euo
#pipefail up top means either one failing kills the script before we get here
touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
