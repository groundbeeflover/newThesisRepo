#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Normal Faster R-CNN baseline run
# -----------------------------

ENV_NAME="DGOD"
RUN_NAME="gwhd_fasterrcnn_baseline_run01"
WEIGHTS_FILE="fasterrcnn_baseline_run01"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting Faster R-CNN baseline run..."
echo "Run directory: ${RUN_DIR}"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

# Create run folders
mkdir -p "${RUN_DIR}/checkpoints"
mkdir -p "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/config_snapshot"

# Save setup snapshot
cp train_GWHD_dgfrcnn.py "${RUN_DIR}/config_snapshot/"
cp fasterrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true

pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt"
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

# Save run command
cat > "${RUN_DIR}/config_snapshot/run_command.txt" <<EOF
python train_GWHD_dgfrcnn.py \\
  --exp non_dg \\
  --weights_folder "${RUN_DIR}/checkpoints" \\
  --weights_file "${WEIGHTS_FILE}" \\
  --reg_weights 1 0.1 1 0.001 0.05
EOF

# Make sure this is a fresh run
if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt" ]; then
  echo "ERROR: Checkpoint already exists:"
  echo "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt"
  echo "Delete it or change RUN_NAME/WEIGHTS_FILE before starting from scratch."
  exit 1
fi

# Start training
python train_GWHD_dgfrcnn.py \
  --exp non_dg \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --reg_weights 1 0.1 1 0.001 0.05 \
  2>&1 | tee "${RUN_DIR}/logs/train.log"