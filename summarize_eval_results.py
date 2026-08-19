#!/usr/bin/env python3
"""
pulls together the map test results from the domain_capped(2) lr sweep that
run_all_evals.sh produces, grouped by learning rate

every run's --eval_map --eval_split test call drops a csv at
<net_root>/<run_name>/<run_name>_test_map.csv, this reads the lot, sticks them
in one table and gives mean +/- std per lr group, so i can compare the groups
and see how much run to run wobble deterministic=False is causing
"""

import argparse
import os
import re

import pandas as pd


def parse_lr_tag(run_name: str) -> str:
    """digs the '1e4' or '1e5' bit out of names like domcap2_bs8_lr1e4_run1"""
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
