#!/usr/bin/env bash
set -euo pipefail

# the ablation grid for the coral bs2-vs-bs8 sampling question, runs one point
# after another since it's a single runpod gpu, unlike the hpc scripts which fan
# out over nodes, each point calls run_gwhd_coral.sh, which trains, evals val and
# test map, and flattens the results into coral_ablation_runs/
#
# seven points, where k is how many distinct domains get forced into a batch and
# samples/domain is batch_size / k:
#
#   capped2_bs2   k=2 samples/domain=1   batches/epoch=1828  <- the control,
#                 matches every historical "bs2 coral" run in coral_runs/
#   diverse_bs2   k=2 samples/domain=1   batches/epoch=1828  <- basically a
#                 duplicate of capped2_bs2 on purpose, domain_diverse at
#                 batch_size=2 collapses to the same k, samples per domain and
#                 steps per epoch as domain_capped(domains_per_batch=2) at
#                 batch_size=2, the only actual code difference is one extra
#                 rng.shuffle(batch) in DomainCappedBatchSampler that
#                 domain_diverse doesn't do, which nudges the rng stream for
#                 later batches without changing the sampling distribution, it's
#                 in here as a rough same seed rng noise floor, not a real
#                 mechanism ablation, so don't expect it to move map any more
#                 than run to run variance would
#   natural_bs2   no domain cap at all   batches/epoch=1828  <- the cleanest
#                 oversampling test in the grid, same batch_size and same
#                 steps/epoch as capped2_bs2, so the only thing that moves is
#                 uniform domain balanced oversampling (on in capped2_bs2) vs
#                 sampling proportional to domain size (off here, apart from the
#                 min 2 domains safeguard), if test_map_50 lands near capped2_bs2
#                 then oversampling isn't the bottleneck, if it's way off then it
#                 is
#   capped2_bs8   k=2 samples/domain=4   batches/epoch=457   <- the comparison
#                 that was actually asked for, same sampler and same k, only
#                 samples/domain and steps/epoch change against capped2_bs2
#   diverse_bs8   k=8 samples/domain=1   batches/epoch=457   <- isolates k going
#                 2->8 while samples/domain stays at 1 vs capped2_bs2, and while
#                 steps/epoch stays fixed vs capped2_bs8
#   capped4_bs8   k=4 samples/domain=2   batches/epoch=457   <- the midpoint, if
#                 the degradation tracks k (2->4->8) rather than samples/domain
#                 (4->2->1), this is the point that separates them
#   natural_bs8   no domain cap at all   batches/epoch=457   <- the bs8
#                 reference, what happens with no artificial domain balancing at
#                 all, pairs with natural_bs2 to check the "oversampling doesn't
#                 matter much" read still holds at bs8
#
# usage:
#   ./run_coral_sampler_ablation.sh [lr]
#
# lr defaults to 1e-4, matching most of the historical coral_runs/ sweep
# safe to interrupt and re-run, each point skips training (not eval) if its
# checkpoint is already there
#
# cost: the three bs2 points each run 1828 steps/epoch against 457 for the four
# bs8 ones, so roughly 4x the wall clock per epoch, budget runpod time for that,
# or comment out diverse_bs2 below if you'd rather skip the low information point
# and keep the gpu hours

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
