#!/usr/bin/env bash
set -euo pipefail

EXP_NUM="${1:-run01}"
EVAL_AFTER="${2:-map}"   # options: no_eval, map, wada, both

ENV_NAME="${ENV_NAME:-DGOD}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_GWHD_dgfrcnn_paper_repro.py}"

RUN_NAME="gwhd_fasterrcnn_baseline_paper_${EXP_NUM}"
WEIGHTS_FILE="fasterrcnn_baseline_paper_${EXP_NUM}"
RUN_DIR="runs/${RUN_NAME}"

# Paper settings for Faster R-CNN on GWHD.
BATCH_SIZE=2
LR=0.001
WEIGHT_DECAY=0.0005
MAX_EPOCHS=100
NUM_WORKERS=16
REG_WEIGHTS=(0.5 0.5 0.5 0.075 0.0001)

echo "Starting paper-aligned Faster R-CNN baseline run..."
echo "Experiment: ${EXP_NUM}"
echo "Run directory: ${RUN_DIR}"
echo "Evaluation after training: ${EVAL_AFTER}"
echo "Training script: ${TRAIN_SCRIPT}"

if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi

conda activate "${ENV_NAME}"

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

cp "${TRAIN_SCRIPT}" "${RUN_DIR}/config_snapshot/"
cp fasterrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true

pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" 2>/dev/null || true
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

cat > "${RUN_DIR}/config_snapshot/run_command.txt" <<EOF
python "${TRAIN_SCRIPT}" \\
  --exp non_dg \\
  --weights_folder "${RUN_DIR}/checkpoints" \\
  --weights_file "${WEIGHTS_FILE}" \\
  --reg_weights ${REG_WEIGHTS[*]} \\
  --batch_size "${BATCH_SIZE}" \\
  --lr "${LR}" \\
  --weight_decay "${WEIGHT_DECAY}" \\
  --num_workers "${NUM_WORKERS}" \\
  --max_epochs "${MAX_EPOCHS}"
EOF

if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt" ]; then
  echo "ERROR: Checkpoint already exists:"
  echo "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.ckpt"
  echo "Use a different experiment number or delete the checkpoint."
  exit 1
fi

python "${TRAIN_SCRIPT}" \
  --exp non_dg \
  --weights_folder "${RUN_DIR}/checkpoints" \
  --weights_file "${WEIGHTS_FILE}" \
  --reg_weights "${REG_WEIGHTS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --max_epochs "${MAX_EPOCHS}" \
  2>&1 | tee "${RUN_DIR}/logs/train.log"

if [ "${EVAL_AFTER}" = "map" ] || [ "${EVAL_AFTER}" = "both" ]; then
  echo "Starting paper-style mAP@50 test evaluation for Faster R-CNN baseline run..."

  cat > "${RUN_DIR}/config_snapshot/map_command.txt" <<EOF
python "${TRAIN_SCRIPT}" \\
  --exp non_dg \\
  --weights_folder "${RUN_DIR}/checkpoints" \\
  --weights_file "${WEIGHTS_FILE}" \\
  --reg_weights ${REG_WEIGHTS[*]} \\
  --batch_size "${BATCH_SIZE}" \\
  --lr "${LR}" \\
  --weight_decay "${WEIGHT_DECAY}" \\
  --eval_map \\
  --eval_split test \\
  --map_output_dir "${RUN_DIR}/map_test"
EOF

  python "${TRAIN_SCRIPT}" \
    --exp non_dg \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --reg_weights "${REG_WEIGHTS[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --eval_map \
    --eval_split test \
    --map_output_dir "${RUN_DIR}/map_test" \
    2>&1 | tee "${RUN_DIR}/logs/map_test.log"

  echo "mAP@50 summary:"
  cat "${RUN_DIR}/map_test/map_summary.csv"
fi

if [ "${EVAL_AFTER}" = "wada" ] || [ "${EVAL_AFTER}" = "both" ]; then
  echo "Starting auxiliary WADA/ADA-style test evaluation for Faster R-CNN baseline run..."

  python "${TRAIN_SCRIPT}" \
    --exp non_dg \
    --weights_folder "${RUN_DIR}/checkpoints" \
    --weights_file "${WEIGHTS_FILE}" \
    --reg_weights "${REG_WEIGHTS[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --eval_wada \
    --eval_split test \
    --score_threshold 0.5 \
    --iou_threshold 0.5 \
    --wada_output_dir "${RUN_DIR}/wada_test" \
    2>&1 | tee "${RUN_DIR}/logs/wada_test.log"

  echo "WADA summary:"
  cat "${RUN_DIR}/wada_test/wada_summary.csv"
fi

if [ "${EVAL_AFTER}" = "no_eval" ]; then
  echo "Skipping post-training evaluation."
fi
