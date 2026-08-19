#!/usr/bin/env bash
set -euo pipefail

# multi seed confirmation runs for the top two sampler configs out of the coral
# sampling ablation (see coral_ablation_runs/sampler_ablation_summary.csv and
# coral_ablation_results_for_supervisor/SUMMARY.md), all 7 ablation points ran a
# single seed (42) at lr=1e-4, ranked by test map@50:
#
#   diverse_bs8   domain_diverse  k=8  samples/domain=1   0.633  best
#   capped2_bs8   domain_capped   k=2  samples/domain=4   0.613  2nd
#   diverse_bs2   domain_diverse  k=2  samples/domain=1   0.589
#   capped4_bs8   domain_capped   k=4  samples/domain=2   0.588
#   capped2_bs2   domain_capped   k=2  samples/domain=1   0.585  historical baseline
#   natural_bs8   natural         --   --                 0.568
#   natural_bs2   natural         --   --                 0.544  worst
#
# the caveat SUMMARY.md raises itself: this project's historical baseline seed
# sweep showed 0.05 to 0.07 test map50 spread on an identical config, and
# diverse_bs8 and capped2_bs8 are only about 0.02 apart, so on one seed each
# they're inside the noise, this script is the obvious next step, 3 seeds (0,1,2)
# per config
#
# unlike the original ablation grid this uses the same hyperparameters as
# run_repro_baseline.sh, run_repro_dgkarthik.sh and run_repro_coral.sh rather
# than the ablation's lr=1e-4, so the results here compare directly against those
# deterministic runs and not just against each other, lr=1e-5, bs=8,
# --num_workers 16, seeds 0/1/2, fully deterministic, reg_weights stays on this
# repo's usual coral convention (0.5 0.5 0.5 0.075 0.0001), only --sampler and
# --domains_per_batch change between the two configs, which is what the ablation
# was varying anyway
#
# same runs/gwhd_repro_coral_top2_<tag>_lr<LR>/ layout, checkpoint naming and
# config_snapshot/.done conventions as run_repro_coral.sh, one directory per tag
# (diverse_bs8, capped2_bs8) holding all 3 seeds
#
# unlike sbatch on the cluster nothing here detaches from your terminal, it's
# just a foreground bash process, run it in tmux or screen (or nohup) so an ssh
# drop doesn't kill it:
#   tmux new -s coral_top2_confirm
#   bash run_coral_top2_seed_confirm.sh
#   #ctrl-b d to detach, tmux attach -t coral_top2_confirm to come back
#
# usage: bash run_coral_top2_seed_confirm.sh [lr]
#   lr defaults to 1e-5, matching the other deterministic repros
#
# if you get torch.OutOfMemoryError, same two levers as the other run_repro_*
# scripts and both are env vars, PHYS_BATCH/ACCUM_STEPS splits the effective bs=8
# into smaller accumulated physical steps (keep PHYS_BATCH * ACCUM_STEPS = 8),
# and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, already exported below,
# handles allocator fragmentation:
#
#   PHYS_BATCH=4 ACCUM_STEPS=2 bash run_coral_top2_seed_confirm.sh
#
# fine to re-run, any (tag, seed) with a .done sentinel skips training, and
# train_GWHD_coralfrcnn.py resumes an in progress seed off last.ckpt first
# anyway, --eval_map still runs either way so the val/test map csvs the
# aggregator reads always get rebuilt

LR="${1:-1e-5}"
ENV_NAME="DGOD"
REG_WEIGHTS=(0.5 0.5 0.5 0.075 0.0001)
SEEDS=(0 1 2)
PHYS_BATCH="${PHYS_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

TAGS=(diverse_bs8 capped2_bs8)
SAMPLERS=(domain_diverse domain_capped)
DOMAINS_PER_BATCH=(8 2)

RESULTS_DIR="coral_top2_seed_runs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Confirming top-2 CORAL sampler configs (${TAGS[*]}) across seeds ${SEEDS[*]}"
echo "lr=${LR}, reg_weights=${REG_WEIGHTS[*]}, physical batch=${PHYS_BATCH}, accumulate_grad_batches=${ACCUM_STEPS} (effective batch=$((PHYS_BATCH * ACCUM_STEPS)))"

#use the conda in /workspace if it's there, it survives a pod restart,
#otherwise fall back to the one in home
if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
else
  source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${ENV_NAME}"

mkdir -p "${RESULTS_DIR}"

for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"
  SAMPLER="${SAMPLERS[$i]}"
  DPB="${DOMAINS_PER_BATCH[$i]}"

  RUN_NAME="gwhd_repro_coral_top2_${TAG}_lr${LR}"
  RUN_DIR="runs/${RUN_NAME}"

  mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/config_snapshot"

  cp train_GWHD_coralfrcnn.py "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
  cp -r dg "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
  cp requirements.txt "${RUN_DIR}/config_snapshot/" 2>/dev/null || true
  pip freeze > "${RUN_DIR}/config_snapshot/pip_freeze.txt"
  conda list > "${RUN_DIR}/config_snapshot/conda_list.txt"
  nvidia-smi > "${RUN_DIR}/config_snapshot/nvidia_smi.txt" || true
  git rev-parse HEAD > "${RUN_DIR}/config_snapshot/git_commit.txt" 2>/dev/null || true

  for SEED in "${SEEDS[@]}"; do
    WEIGHTS_FILE="coral_seed${SEED}"
    RUN_TAG="${TAG}_seed${SEED}"

    echo
    echo "=== ${TAG} seed ${SEED}: sampler=${SAMPLER} domains_per_batch=${DPB} ==="

    if [ -f "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done" ]; then
      echo "Seed ${SEED} already completed (found ${WEIGHTS_FILE}.done). Skipping training."
    else
      #train_GWHD_coralfrcnn.py checkpoints every epoch and picks back up from
      #last.ckpt if this seed got interrupted, say a spot pod got reclaimed, so
      #re-running this script is always safe
      python train_GWHD_coralfrcnn.py \
        --exp coral \
        --weights_folder "${RUN_DIR}/checkpoints" \
        --weights_file "${WEIGHTS_FILE}" \
        --sampler "${SAMPLER}" \
        --domains_per_batch "${DPB}" \
        --reg_weights "${REG_WEIGHTS[@]}" \
        --lr "${LR}" \
        --batch_size "${PHYS_BATCH}" \
        --accumulate_grad_batches "${ACCUM_STEPS}" \
        --num_workers 16 \
        --seed "${SEED}" \
        --deterministic \
        2>&1 | tee "${RUN_DIR}/logs/train_seed${SEED}.log"

      #train_GWHD_coralfrcnn.py doesn't write its own WEIGHTS_FILE.done
      #sentinel the way the baseline and mattia scripts do, so write one here
      #and the skip-if-done check above behaves the same
      touch "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}.done"
    fi

    echo "-- Evaluating val + test for ${RUN_TAG} --"
    for split in val test; do
      python train_GWHD_coralfrcnn.py \
        --exp coral \
        --eval_map \
        --eval_split "${split}" \
        --weights_folder "${RUN_DIR}/checkpoints" \
        --weights_file "${WEIGHTS_FILE}" \
        2>&1 | tee "${RUN_DIR}/logs/${split}_eval_seed${SEED}.log"
    done

    cp "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}_val_map.csv" "${RESULTS_DIR}/${RUN_TAG}_val_map.csv" 2>/dev/null || echo "!! missing val_map.csv for ${RUN_TAG}"
    cp "${RUN_DIR}/checkpoints/${WEIGHTS_FILE}_test_map.csv" "${RESULTS_DIR}/${RUN_TAG}_test_map.csv" 2>/dev/null || echo "!! missing test_map.csv for ${RUN_TAG}"
  done
done

echo
echo "Done (or already were). Check runs/gwhd_repro_coral_top2_*_lr${LR}/checkpoints/*.done to confirm which (tag, seed) pairs finished."

echo
echo "=== Aggregating mean/std across seeds: python summarize_coral_top2_seed_confirm.py ==="
python summarize_coral_top2_seed_confirm.py \
  --results_dir "${RESULTS_DIR}" \
  --tags "${TAGS[@]}" \
  --seeds "${SEEDS[@]}" \
  --out_csv "${RESULTS_DIR}/top2_seed_confirm_summary.csv"
