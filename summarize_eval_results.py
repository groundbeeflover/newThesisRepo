#!/usr/bin/env python3
"""
Aggregate mAP test results across the domain_capped(2) LR-sweep runs
produced by run_all_evals.sh, grouped by learning rate.

Each run's `--eval_map --eval_split test` call writes a CSV at
<net_root>/<run_name>/<run_name>_test_map.csv (see evaluate_map's
output_path in train_GWHD_coralfrcnn.py). This script reads all of those,
combines them into one table, and reports mean +/- std per LR group so you
can compare the two groups (and check within-group run-to-run variance from
deterministic=False) directly.
"""

import argparse
import os
import re

import pandas as pd


def parse_lr_tag(run_name: str) -> str:
    """Pulls '1e4' / '1e5' etc. out of names like domcap2_bs8_lr1e4_run1."""
    match = re.search(r"lr(1e\d+)", run_name)
    return match.group(1) if match else "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net_root",
        default="coral_runs",
        help="Directory containing the per-run subfolders.",
    )
    parser.add_argument(
        "--run_names",
        nargs="+",
        required=True,
        help="Run names, e.g. domcap2_bs8_lr1e4_run1 domcap2_bs8_lr1e4_run2 ...",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional path to save the combined per-run table.",
    )
    args = parser.parse_args()

    rows = []
    for run_name in args.run_names:
        csv_path = os.path.join(
            args.net_root, run_name, f"{run_name}_test_map.csv"
        )
        if not os.path.exists(csv_path):
            print(f"!! Missing results for {run_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        row = df.iloc[0].to_dict()
        row["run_name"] = run_name
        row["lr_tag"] = parse_lr_tag(run_name)
        rows.append(row)

    if not rows:
        raise SystemExit(
            "No result CSVs found. Did run_all_evals.sh finish successfully?"
        )

    combined = pd.DataFrame(rows)
    metric_cols = ["map_50", "map_mean_iou_10_50_75", "map_75"]
    combined = combined[["run_name", "lr_tag"] + metric_cols]

    print("\nPer-run test results")
    print("=====================")
    print(combined.to_string(index=False))

    print("\nGrouped by learning rate (mean +/- std)")
    print("========================================")
    grouped = combined.groupby("lr_tag")[metric_cols].agg(["mean", "std"])
    print(grouped.to_string())

    if args.out_csv:
        combined.to_csv(args.out_csv, index=False)
        print(f"\nSaved combined per-run table to {args.out_csv}")


if __name__ == "__main__":
    main()
