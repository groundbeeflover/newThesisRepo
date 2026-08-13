#!/usr/bin/env bash
set -euo pipefail

# Generalized CORAL training run for the bs2-vs-bs8 sampling-bottleneck
# ablation (see run_coral_sampler_ablation.sh for the full grid this feeds).
#
# Usage:
#   ./run_gwhd_coral.sh <tag> <sampler> <batch_size> <domains_per_batch> [lr] [wada]
#
#   sampler:            domain_diverse | domain_capped | natural
#   domains_per_batch:  only affects behaviour when sampler=domain_capped
#                        (pass anything, e.g. 2, for domain_diverse/natural)
#
# Examples (the 5-point grid from run_coral_sampler_ablation.sh):
#   ./run_gwhd_coral.sh capped2_bs2  domain_capped  2 2 1e-4   # control: K=2, samples/domain=1
#   ./run_gwhd_coral.sh capped2_bs8  domain_capped  8 2 1e-4   # the comparison you asked for: K=2, samples/domain=4
#   ./run_gwhd_coral.sh diverse_bs8  domain_diverse 8 8 1e-4   # isolates K: K=8, samples/domain=1
#   ./run_gwhd_coral.sh capped4_bs8  domain_capped  8 4 1e-4   # midpoint: K=4, samples/domain=2
#   ./run_gwhd_coral.sh natural_bs8  natural         8 2 1e-4  # reference: no domain balancing
#
# Each run writes to runs/gwhd_coral_ablation_<tag>/ (checkpoints, logs,
# config snapshot, domain-histogram + coral-diag CSVs from the
# on_train_epoch_end instrumentation in train_GWHD_coralfrcnn.py), then also
# copies its val/test mAP + those two diagnostic CSVs, flattened, into
# coral_ablation_runs/ for summarize_coral_ablation.py to pick up.
#
# Safe to re-run: skips training (but not eval) if the checkpoint already
# exists, matching the other run_gwhd_*.sh scripts' convention.

TAG="${1:?Usage: run_gwhd_coral.sh <tag> <sampler> <batch_size> <domains_per_batch> [lr] [wada]}"
SAMPLER="${2:?sampler required: domain_diverse|domain_capped|natural}"
BATCH_SIZE="${3:?batch_size required}"
DOMAINS_PER_BATCH="${4:?domains_per_batch required (ignored unless sampler=domain_capped)}"
LR="${5:-0.0001}"
RUN_WADA="${6:-no_wada}"

ENV_NAME="DGOD"
RUN_NAME="gwhd_coral_ablation_${TAG}"
WEIGHTS_FILE="coral_ablation_${TAG}"
RUN_DIR="runs/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"
RESULTS_DIR="coral_ablation_runs"

echo "Starting CORAL sampling-ablation run..."
echo "Tag=${TAG} sampler=${SAMPLER} batch_size=${BATCH_SIZE} domains_per_batch=${DOMAINS_PER_BATCH} lr=${LR}"
echo "Run directory: ${RUN_DIR}"

# Prefer persistent conda if installed in /workspace; fall back to ~/miniconda3.
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi

conda activate "${ENV_NAME}"

mkdir -p "${CKPT_DIR}" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot" "${RESULTS_DIR}"

cp train_GWHD_coralfrcnn.py "${RUN_DIR}/config_snapshot/"
cp -r dg "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true

pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt"
git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

cat > "${RUN_DIR}/config_snapshot/run_command.txt" <<EOF
python train_GWHD_coralfrcnn.py \\
  --exp coral \\
  --weights_folder "${CKPT_DIR}" \\
  --weights_file "${WEIGHTS_FILE}" \\
  --sampler ${SAMPLER} \\
  --batch_size ${BATCH_SIZE} \\
  --domains_per_batch ${DOMAINS_PER_BATCH} \\
  --lr ${LR} \\
  --reg_weights 0.5 0.5 0.5 0.075 0.0001
EOF

# Record the config axes explicitly, since they're the whole point of this
# grid -- summarize_coral_ablation.py reads this instead of re-parsing tags.
cat > "${RESULTS_DIR}/${TAG}_config.json" <<EOF
{
  "tag": "${TAG}",
  "sampler": "${SAMPLER}",
  "batch_size": ${BATCH_SIZE},
  "domains_per_batch": ${DOMAINS_PER_BATCH},
  "lr": ${LR}
}
EOF

if [ -f "${CKPT_DIR}/${WEIGHTS_FILE}.ckpt" ]; then
  echo "Checkpoint already exists: ${CKPT_DIR}/${WEIGHTS_FILE}.ckpt -- skipping training, running eval only."
else
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --weights_folder "${CKPT_DIR}" \
    --weights_file "${WEIGHTS_FILE}" \
    --sampler "${SAMPLER}" \
    --batch_size "${BATCH_SIZE}" \
    --domains_per_batch "${DOMAINS_PER_BATCH}" \
    --lr "${LR}" \
    --reg_weights 0.5 0.5 0.5 0.075 0.0001 \
    2>&1 | tee "${RUN_DIR}/logs/train.log"
fi

echo "== Evaluating val + test =="
for split in val test; do
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --eval_map \
    --eval_split "${split}" \
    --weights_folder "${CKPT_DIR}" \
    --weights_file "${WEIGHTS_FILE}" \
    2>&1 | tee "${RUN_DIR}/logs/${split}_eval.log"
done

echo "== Flattening results into ${RESULTS_DIR}/ =="
cp "${CKPT_DIR}/${WEIGHTS_FILE}_val_map.csv" "${RESULTS_DIR}/${TAG}_val_map.csv" 2>/dev/null || echo "!! missing val_map.csv"
cp "${CKPT_DIR}/${WEIGHTS_FILE}_test_map.csv" "${RESULTS_DIR}/${TAG}_test_map.csv" 2>/dev/null || echo "!! missing test_map.csv"
cp "${CKPT_DIR}/${WEIGHTS_FILE}_domain_histogram.csv" "${RESULTS_DIR}/${TAG}_domain_histogram.csv" 2>/dev/null || echo "!! missing domain_histogram.csv"
cp "${CKPT_DIR}/${WEIGHTS_FILE}_coral_diag.csv" "${RESULTS_DIR}/${TAG}_coral_diag.csv" 2>/dev/null || echo "!! missing coral_diag.csv"

if [ "${RUN_WADA}" = "wada" ] || [ "${RUN_WADA}" = "--wada" ] || [ "${RUN_WADA}" = "true" ]; then
  echo "Starting WADA test evaluation..."
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --weights_folder "${CKPT_DIR}" \
    --weights_file "${WEIGHTS_FILE}" \
    --eval_wada \
    --eval_split test \
    --score_threshold 0.5 \
    --iou_threshold 0.5 \
    --wada_output_dir "${RUN_DIR}/wada_test" \
    2>&1 | tee "${RUN_DIR}/logs/wada_test.log"

  cp "${RUN_DIR}/wada_test/wada_summary.csv" "${RESULTS_DIR}/${TAG}_wada_summary.csv" 2>/dev/null || true
fi

echo "Done: ${TAG}"
