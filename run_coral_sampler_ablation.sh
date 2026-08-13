#!/usr/bin/env bash
set -euo pipefail

# Bottleneck-isolation ablation grid for the CORAL bs2-vs-bs8 sampling
# question (Aug 2026 thesis debugging). Runs sequentially -- assumes a single
# RunPod GPU, unlike the HPC scripts which fan out across nodes. Each point
# calls run_gwhd_coral.sh, which trains + evals (val/test mAP) + flattens
# results into coral_ablation_runs/.
#
# Seven points (K = distinct domains forced per batch, samples/domain =
# batch_size / K):
#
#   capped2_bs2   K=2 samples/domain=1   batches/epoch=1828  <- control,
#                 matches every historical "bs2 CORAL" run in coral_runs/
#   diverse_bs2   K=2 samples/domain=1   batches/epoch=1828  <- near-duplicate
#                 of capped2_bs2 by design (domain_diverse at batch_size=2
#                 degenerates to the same K/samples-per-domain/steps-per-epoch
#                 as domain_capped(domains_per_batch=2) at batch_size=2). The
#                 only code-level difference is one extra rng.shuffle(batch)
#                 call in DomainCappedBatchSampler that domain_diverse doesn't
#                 do, which shifts the RNG stream for later batches but not
#                 the underlying sampling distribution. Included as a rough
#                 same-seed RNG-noise floor, not a real mechanism ablation --
#                 don't expect this one to move mAP by more than run-to-run
#                 variance would anyway.
#   natural_bs2   no domain cap at all   batches/epoch=1828  <- the cleanest
#                 oversampling test in the whole grid: same batch_size and
#                 same steps/epoch as capped2_bs2, so the *only* thing that
#                 changes is uniform domain-balanced oversampling (on, in
#                 capped2_bs2) vs proportional-to-domain-size natural sampling
#                 (off, here, modulo the min-2-domains safeguard). If
#                 test_map_50 here is close to capped2_bs2, oversampling
#                 itself isn't the bottleneck; if it's very different, it is.
#   capped2_bs8   K=2 samples/domain=4   batches/epoch=457   <- the original
#                 comparison requested: same sampler, same K, only
#                 samples/domain and steps/epoch change vs capped2_bs2
#   diverse_bs8   K=8 samples/domain=1   batches/epoch=457   <- isolates K
#                 (2->8) while holding samples/domain=1 fixed vs capped2_bs2,
#                 and holding steps/epoch fixed vs capped2_bs8
#   capped4_bs8   K=4 samples/domain=2   batches/epoch=457   <- midpoint;
#                 if degradation tracks K monotonically (2->4->8) vs tracking
#                 samples/domain (4->2->1), this point tells them apart
#   natural_bs8   no domain cap at all   batches/epoch=457   <- reference at
#                 bs8: what happens with no artificial domain balancing: pairs
#                 with natural_bs2 to check whether the "oversampling doesn't
#                 matter much" read from bs2 also holds at bs8
#
# Usage:
#   ./run_coral_sampler_ablation.sh [lr]
#
# lr defaults to 1e-4, matching most of the historical coral_runs/ sweep.
# Safe to interrupt and re-run: each point skips training (not eval) if its
# checkpoint already exists.
#
# Cost note: the three bs2 points (capped2_bs2, diverse_bs2, natural_bs2) each
# run 1828 steps/epoch vs 457 for the four bs8 points -- roughly 4x the
# wall-clock per epoch. Budget RunPod time accordingly, or comment out
# diverse_bs2 below if you'd rather skip the low-information point and save
# the GPU hours.

LR="${1:-0.0001}"

echo "=== [1/7] capped2_bs2  -- control point (K=2, samples/domain=1) ==="
./run_gwhd_coral.sh capped2_bs2 domain_capped 2 2 "${LR}"

echo "=== [2/7] diverse_bs2  -- near-duplicate of capped2_bs2, RNG-noise floor ==="
./run_gwhd_coral.sh diverse_bs2 domain_diverse 2 2 "${LR}"

echo "=== [3/7] natural_bs2  -- cleanest oversampling isolation (same batch_size/steps as capped2_bs2) ==="
./run_gwhd_coral.sh natural_bs2 natural 2 2 "${LR}"

echo "=== [4/7] capped2_bs8  -- the original comparison requested (K=2, samples/domain=4) ==="
./run_gwhd_coral.sh capped2_bs8 domain_capped 8 2 "${LR}"

echo "=== [5/7] diverse_bs8  -- isolates domain-count K (K=8, samples/domain=1) ==="
./run_gwhd_coral.sh diverse_bs8 domain_diverse 8 8 "${LR}"

echo "=== [6/7] capped4_bs8  -- midpoint on the K vs samples/domain axis (K=4, samples/domain=2) ==="
./run_gwhd_coral.sh capped4_bs8 domain_capped 8 4 "${LR}"

echo "=== [7/7] natural_bs8  -- reference at bs8: no domain balancing beyond the CORAL minimum ==="
./run_gwhd_coral.sh natural_bs8 natural 8 2 "${LR}"

echo
echo "=== Aggregating: python summarize_coral_ablation.py ==="
python summarize_coral_ablation.py \
  --results_dir coral_ablation_runs \
  --tags capped2_bs2 diverse_bs2 natural_bs2 capped2_bs8 diverse_bs8 capped4_bs8 natural_bs8 \
  --out_csv coral_ablation_runs/sampler_ablation_summary.csv
