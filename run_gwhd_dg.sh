#!/usr/bin/env bash
set -euo pipefail

EXP_NUM="${1:-run01}"

ENV_NAME="DGOD"
RUN_NAME="gwhd_dgfrcnn_dg_paperweights_${EXP_NUM}"
WEIGHTS_FILE="dgfrcnn_dg_paperweights_${EXP_NUM}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting DG Faster R-CNN run..."
echo "Experiment: ${EXP_NUM}"
echo "Run directory: ${RUN_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

cp train_GWHD_dgfrcnn.py "${RUN_DIR}/config_snapshot/"
cp fasterrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true

pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt"
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

cat > "${RUN_DIR}/config_snapshot/run_command.txt" <<EOF
python train_GWHD_dgfrcnn.py \\
  --exp dg \\
  --weights_folder "${RUN_DIR}/checkpoints" \\
  --weights_file "${WEIGHTS_FILE}" \\
  --reg_weights 1 0.1 1 0.001 0.05
EOF

if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt" ]; then
  echo "ERROR: Checkpoint already exists:"
  echo "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt"
  echo "Use a different experiment number or delete the checkpoint."
  exit 1
fi

python train_GWHD_dgfrcnn.py \
  --exp dg \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --reg_weights 1 0.1 1 0.001 0.05 \
  2>&1 | tee "${RUN_DIR}/logs/train.log"