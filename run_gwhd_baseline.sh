#!/usr/bin/env bash
set -euo pipefail

EXP_NUM="${1:-run01}"
RUN_WADA="${2:-no_wada}"

ENV_NAME="DGOD"
RUN_NAME="gwhd_fasterrcnn_baseline_${EXP_NUM}"
WEIGHTS_FILE="fasterrcnn_baseline_${EXP_NUM}"
RUN_DIR="runs/${RUN_NAME}"

echo "Starting Faster R-CNN baseline run..."
echo "Experiment: ${EXP_NUM}"
echo "Run directory: ${RUN_DIR}"
echo "Run WADA after training: ${RUN_WADA}"

# Prefer persistent conda if installed in /workspace; fall back to ~/miniconda3.
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi

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
  --exp non_dg \\
  --weights_folder "${RUN_DIR}/checkpoints" \\
  --weights_file "${WEIGHTS_FILE}" \\
  --reg_weights 0.5 0.5 0.5 0.075 0.0001
EOF

if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt" ]; then
  echo "ERROR: Checkpoint already exists:"
  echo "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt"
  echo "Use a different experiment number or delete the checkpoint."
  exit 1
fi

python train_GWHD_dgfrcnn.py \
  --exp non_dg \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --reg_weights 0.5 0.5 0.5 0.075 0.0001 \
  2>&1 | tee "${RUN_DIR}/logs/train.log"

if [ "${RUN_WADA}" = "wada" ] || [ "${RUN_WADA}" = "--wada" ] || [ "${RUN_WADA}" = "true" ]; then
  echo "Starting WADA test evaluation for Faster R-CNN baseline run..."

  cat > "${RUN_DIR}/config_snapshot/wada_command.txt" <<EOF
python train_GWHD_dgfrcnn.py \\
  --exp non_dg \\
  --weights_folder "${RUN_DIR}/checkpoints" \\
  --weights_file "${WEIGHTS_FILE}" \\
  --reg_weights 0.5 0.5 0.5 0.075 0.0001 \\
  --eval_wada \\
  --eval_split test \\
  --score_threshold 0.5 \\
  --iou_threshold 0.5 \\
  --wada_output_dir "${RUN_DIR}/wada_test"
EOF

  python train_GWHD_dgfrcnn.py \
    --exp non_dg \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --reg_weights 0.5 0.5 0.5 0.075 0.0001 \
    --eval_wada \
    --eval_split test \
    --score_threshold 0.5 \
    --iou_threshold 0.5 \
    --wada_output_dir "${RUN_DIR}/wada_test" \
    2>&1 | tee "${RUN_DIR}/logs/wada_test.log"

  echo "WADA summary:"
  cat "${RUN_DIR}/wada_test/wada_summary.csv"
else
  echo "Skipping WADA evaluation. To run it automatically, use:"
  echo "./run_gwhd_baseline.sh ${EXP_NUM} wada"
fi