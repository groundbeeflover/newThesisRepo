#!/usr/bin/env bash
# runs --eval_map --eval_split test over every checkpoint the domain_capped(2)
# lr sweep produced, then aggregates the lot
#
# expects the 6 checkpoints from the earlier training command to already be
# sitting at ${NET_ROOT}/<run_name>/<run_name>.ckpt, launch it from wherever
# train_GWHD_coralfrcnn.py resolves its data/Annots/... paths from, normally the
# newThesisRepo root, and change NET_ROOT below if coral_runs/ lives somewhere
# else relative to that

set -euo pipefail

NET_ROOT="GWHD_CORAL"

RUN_NAMES=(
  domcap2_bs8_lr1e4_run0
  domcap2_bs8_lr1e4_run1
  domcap2_bs8_lr1e4_run2
  domcap2_bs8_lr1e5_run0
  domcap2_bs8_lr1e5_run1
  domcap2_bs8_lr1e5_run2
)

for run_name in "${RUN_NAMES[@]}"; do
  ckpt="${NET_ROOT}/${run_name}/${run_name}.ckpt"

  if [[ ! -f "${ckpt}" ]]; then
    echo "!! Skipping ${run_name}: no checkpoint found at ${ckpt}"
    continue
  fi

  echo "== Evaluating ${run_name} on test split =="
  python train_GWHD_coralfrcnn.py \
    --exp coral \
    --eval_map \
    --eval_split test \
    --weights_folder "${NET_ROOT}/${run_name}" \
    --weights_file "${run_name}" \
    2>&1 | tee "${NET_ROOT}/${run_name}_test_eval.log"
  echo
done

echo "== Aggregating results =="
python summarize_eval_results.py \
  --net_root "${NET_ROOT}" \
  --run_names "${RUN_NAMES[@]}" \
  --out_csv "${NET_ROOT}/domcap2_bs8_lr_sweep_summary.csv"
