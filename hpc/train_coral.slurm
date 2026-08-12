#!/bin/bash
#SBATCH --job-name=gwhd_coral
#SBATCH --output=hpc_logs/stdout-%x-%j.out
#SBATCH --error=hpc_logs/stderr-%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=education
#SBATCH --account=education

# Usage: sbatch hpc/train_coral.slurm <exp: coral|non_dg> <tag>
# Example: sbatch hpc/train_coral.slurm coral run01
set -euo pipefail

EXP="${1:-coral}"
TAG="${2:-run01}"
RUN_NAME="gwhd_${EXP}_${TAG}"
RUN_DIR="runs/${RUN_NAME}"

mkdir -p hpc_logs "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

echo "Job ${SLURM_JOB_ID} on $(hostname), exp=${EXP}, tag=${TAG}, run_dir=${RUN_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DGOD

cp train_GWHD_coralfrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

WEIGHTS_FILE="${RUN_NAME}"

if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt" ]; then
  echo "ERROR: Checkpoint already exists: ${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt"
  echo "Use a different tag or delete the checkpoint."
  exit 1
fi

# NOTE: launched with srun so PyTorch Lightning's auto-detected SLURMEnvironment
# plugin actually gets the handshake it expects (see hpc/README.md, section 4).
srun python train_GWHD_coralfrcnn.py \
  --exp "${EXP}" \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --reg_weights 0.5 0.5 0.5 0.075 0.0001 \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  2>&1 | tee "${RUN_DIR}/logs/train.log"
