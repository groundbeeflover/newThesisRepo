#!/usr/bin/env python3
"""
Aggregate the multi-seed confirmation runs produced by
run_coral_top2_seed_confirm.sh for the two leading CORAL sampler configs from
the bs2-vs-bs8 sampling ablation (diverse_bs8, capped2_bs8 -- see that
script's header for the full ranking and where the numbers came from).

For each tag, reads (from --results_dir, default coral_top2_seed_runs/):
  <tag>_seed<N>_val_map.csv    evaluate_map() output, val split
  <tag>_seed<N>_test_map.csv   evaluate_map() output, test split

and prints/saves one row per tag with the mean/std of val and test mAP@50
across seeds, so "does the ablation's single-seed ranking survive seed
variance" can be read off directly.
"""

import argparse
import os

import pandas as pd


def load_map(results_dir: str, tag: str, seed: int, split: str) -> dict:
    path = os.path.join(results_dir, f"{tag}_seed{seed}_{split}_map.csv")
    if not os.path.exists(path):
        return {f"{split}_map_50": None, f"{split}_map_mean": None}
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {
        f"{split}_map_50": row["map_50"],
        f"{split}_map_mean": row["map_mean_iou_10_50_75"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="coral_top2_seed_runs")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    per_seed_rows = []
    for tag in args.tags:
        for seed in args.seeds:
            row = {"tag": tag, "seed": seed}
            row.update(load_map(args.results_dir, tag, seed, "val"))
            row.update(load_map(args.results_dir, tag, seed, "test"))
            per_seed_rows.append(row)

    per_seed = pd.DataFrame(per_seed_rows)
    missing = per_seed[per_seed["test_map_50"].isna()]
    if not missing.empty:
        print("!! Missing results for:")
        print(missing[["tag", "seed"]].to_string(index=False))

    present = per_seed.dropna(subset=["test_map_50"])
    if present.empty:
        raise SystemExit("No runs found. Did run_coral_top2_seed_confirm.sh finish any (tag, seed) pairs?")

    summary = present.groupby("tag").agg(
        n_seeds=("seed", "count"),
        val_map_50_mean=("val_map_50", "mean"),
        val_map_50_std=("val_map_50", "std"),
        test_map_50_mean=("test_map_50", "mean"),
        test_map_50_std=("test_map_50", "std"),
    ).reset_index()
    summary = summary.sort_values("test_map_50_mean", ascending=False)

    print("\nPer-seed results")
    print("=================")
    print(per_seed.to_string(index=False))

    print("\nAcross-seed summary (sorted by test mAP@50, descending)")
    print("=========================================================")
    print(summary.to_string(index=False))

    if len(summary) >= 2:
        gap = summary.iloc[0]["test_map_50_mean"] - summary.iloc[1]["test_map_50_mean"]
        pooled_std = max(summary.iloc[0]["test_map_50_std"], summary.iloc[1]["test_map_50_std"])
        print(
            f"\nGap between top two configs' mean test mAP@50: {gap:.4f} "
            f"(vs. within-config std of up to {pooled_std:.4f} across {present['seed'].nunique()} seeds). "
            "If the gap is smaller than or comparable to that std, the single-seed ablation's "
            "ranking is not yet confirmed -- treat the configs as statistically tied."
        )

    if args.out_csv:
        summary.to_csv(args.out_csv, index=False)
        per_seed_csv = os.path.splitext(args.out_csv)[0] + "_per_seed.csv"
        per_seed.to_csv(per_seed_csv, index=False)
        print(f"\nSaved summary to {args.out_csv}")
        print(f"Saved per-seed table to {per_seed_csv}")


if __name__ == "__main__":
    main()
