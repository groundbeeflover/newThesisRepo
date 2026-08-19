#!/bin/bash
#SBATCH --job-name=gwhd_smoke
#SBATCH --output=hpc_logs/stdout-%x-%j.out
#SBATCH --error=hpc_logs/stderr-%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --partition=research
#SBATCH --account=research

# quick sanity check before burning a full run, just confirms the venv, cuda and
# the gwhd data path all work, 1 epoch, tiny batch, capped steps
# usage: sbatch hpc/smoke_test.slurm
set -euo pipefail

mkdir -p hpc_logs runs/smoke_hpc/checkpoints runs/smoke_hpc/logs

echo "Smoke test job ${SLURM_JOB_ID} on $(hostname)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DGOD

nvidia-smi

srun python train_GWHD_coralfrcnn.py \
  --exp coral \
  --weights_folder runs/smoke_hpc/checkpoints \
  --weights_file smoke_hpc \
  --reg_weights 0.5 0.5 0.5 0.075 0.0001 \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --max_epochs 1 \
  2>&1 | tee runs/smoke_hpc/logs/smoke.log

echo "Smoke test finished OK."
