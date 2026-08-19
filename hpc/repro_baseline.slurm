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

# reproduction of seemakurthy et al's baseline number using the clean stock
# torchvision faster r-cnn in train_GWHD_baseline_clean.py, NOT the repo's own
# fasterrcnn.py/non_dg path, going off mattia dutto's email, forwarded by petra
# bosilj 2026-08-12: bs=8, lr=1e-5, seeds 0/1/2, fully deterministic
# target from mattia's runs is avg test map@0.5 around 54.7 (see hpc/README.md)
#
# array index 0/1/2 maps to seed 0/1/2, one job each, all running at once
#
# runs on the research partition (MaxTime=7-00:00:00, check with scontrol show
# partition research) rather than education (MaxTime=02:00:00, and its
# AllowAccounts=education locks this account out anyway, see hpc/README.md
# section 6), 24h is a generous one shot budget for 100 epochs with patience 10
# early stopping, the earlier grlbaselineruns baseline converged in 13 to 23
#
# the training script still checkpoints every epoch and picks back up from
# runs/.../checkpoints/last.ckpt if it's there, and still only writes
# WEIGHTS_FILE.done once val/test eval genuinely finishes, that's a safety net
# for a job getting killed, a node rebooting, or hitting a real time limit later
# on, even though 24h shouldn't need it, resubmitting the same command is always
# safe, it resumes whatever's unfinished and skips whatever's already .done
#
# usage: sbatch hpc/repro_baseline.slurm [lr]
#   lr defaults to 1e-5, the one with a known target number, pass 1e-4 to run the
#   other row mattia says is in the sheet (sheet3, rows 3+5), nobody's shared a
#   target for that one yet
# if a task ooms, and l40s have 48gb so it shouldn't, PHYS_BATCH/ACCUM_STEPS are
# here to match run_repro_baseline.sh in case you move a seed over there:
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

#launched with srun so lightning's auto-detected slurm environment plugin
#actually gets the handshake it wants (see hpc/README.md, section 4)
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
