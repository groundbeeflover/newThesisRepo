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

# reproduction of seemakurthy et al's dgkarthik (grl) number using
# train_GWHD_dgfrcnn_mattia.py, which is mattia dutto's script forwarded by petra
# bosilj 2026-08-12, patched here so it runs outside a notebook and resolves its
# own checkpoint path instead of a hardcoded "GWHD/" (see the git diff, or
# hpc/README.md), settings per the email: bs=8, hardcoded in the script itself,
# lr=1e-5, alpha_4=0.055, seeds 0/1/2, fully deterministic
# target from mattia's runs is avg test map@0.5 around 60.3 (see hpc/README.md)
#
# on reg_weights (a b c d e): the email only ever specified alpha_4 (d) as "the
# only parameter of note" = 0.055, a/b/c/e are not confirmed by mattia, so this
# uses this repo's existing convention, 0.5 0.5 0.5 x 0.0001 (see run_gwhd_dg.sh),
# with only d swapped over, update REG_WEIGHTS below if mattia confirms different
# values for the other four
#
# array index 0/1/2 maps to seed 0/1/2, one job each, all running at once
#
# runs on the research partition (MaxTime=7-00:00:00, check with scontrol show
# partition research) rather than education (MaxTime=02:00:00, and its
# AllowAccounts=education locks this account out anyway, see hpc/README.md
# section 6), 24h is a generous one shot budget, this run took 35 to 37 epochs in
# the earlier grl attempts in grlbaselineruns/, which fits inside 24h easily even
# at a few minutes an epoch
#
# the training script still checkpoints every epoch and picks back up from
# runs/.../checkpoints/last.ckpt if it's there, and still only writes
# WEIGHTS_FILE.done once val/test eval genuinely finishes, safety net, even
# though 24h shouldn't need it, resubmitting the same command is always safe, it
# resumes whatever's unfinished and skips whatever's already .done
#
# usage: sbatch hpc/repro_dgkarthik.slurm [lr]
#   lr defaults to 1e-5, the one with a known target number, pass 1e-4 to run the
#   other row mattia says is in the sheet (sheet3, rows 3+5), nobody's shared a
#   target for that one yet
# if a task ooms, and l40s have 48gb so it shouldn't, PHYS_BATCH/ACCUM_STEPS are
# here to match run_repro_dgkarthik.sh in case you move a seed over there:
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

#launched with srun so lightning's auto-detected slurm environment plugin
#actually gets the handshake it wants (see hpc/README.md, section 4)
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
#separate call, separate log, loads the checkpoint that just got trained and
#writes ${WEIGHTS_FILE}_test_map.csv (map/map_50/map_75) into checkpoints/, the
#same two phase thing run_repro_dgkarthik.sh does, without this second call this
#seed never produces a test log or a results csv at all
srun python train_GWHD_dgfrcnn_mattia.py \
  --exp dg \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --eval_map \
  --eval_split test \
  2>&1 | tee "${RUN_DIR}/logs/test_seed${SEED}.log"

#only mark the seed done once training and eval have both worked, set -euo
#pipefail up top means either one failing kills the script before we get here
touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
