#!/usr/bin/env python3
"""
Aggregate the CORAL bs2-vs-bs8 sampling-bottleneck ablation grid produced by
run_coral_sampler_ablation.sh / run_gwhd_coral.sh.

For each run tag, reads (from --results_dir, default coral_ablation_runs/):
  <tag>_config.json           sampler / batch_size / domains_per_batch / lr
  <tag>_val_map.csv           evaluate_map() output, val split
  <tag>_test_map.csv          evaluate_map() output, test split
  <tag>_domain_histogram.csv  per-epoch, per-domain sampler stats (from
                               dg/samplers.py get_last_epoch_stats(), dumped
                               each epoch by DGFRCNN.on_train_epoch_end)
  <tag>_coral_diag.csv        per-epoch means of the pre-cap sample-count
                               diagnostics from FRCNNCoralLosses.forward()

and produces one row per tag with the mechanism axes (K = distinct domains
forced per batch, samples/domain = batch_size / K) alongside the mAP results
and the two bottleneck diagnostics, so the isolation question --
"is it oversampling, K, or per-domain sample starvation?" -- can be read off
directly instead of re-derived from logs.
"""

import argparse
import json
import os

import pandas as pd


def load_config(results_dir: str, tag: str) -> dict:
    path = os.path.join(results_dir, f"{tag}_config.json")
    with open(path) as f:
        cfg = json.load(f)

    sampler = cfg["sampler"]
    batch_size = cfg["batch_size"]
    domains_per_batch = cfg["domains_per_batch"]

    if sampler == "domain_diverse":
        k = batch_size
    elif sampler == "domain_capped":
        k = domains_per_batch
    else:  # natural: no fixed K, domains-per-batch is whatever falls out of chance
        k = None

    samples_per_domain = (batch_size / k) if k else None

    cfg["K_domains_per_batch"] = k
    cfg["samples_per_domain"] = samples_per_domain
    return cfg


def load_map(results_dir: str, tag: str, split: str) -> dict:
    path = os.path.join(results_dir, f"{tag}_{split}_map.csv")
    if not os.path.exists(path):
        return {f"{split}_map_50": None, f"{split}_map_mean": None}
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {
        f"{split}_map_50": row["map_50"],
        f"{split}_map_mean": row["map_mean_iou_10_50_75"],
    }


def load_domain_histogram_summary(results_dir: str, tag: str) -> dict:
    path = os.path.join(results_dir, f"{tag}_domain_histogram.csv")
    if not os.path.exists(path):
        return {
            "oversample_ratio_smallest_domain_final_epoch": None,
            "oversample_ratio_max_final_epoch": None,
        }
    df = pd.read_csv(path)
    if df.empty:
        return {
            "oversample_ratio_smallest_domain_final_epoch": None,
            "oversample_ratio_max_final_epoch": None,
        }

    final_epoch = df["epoch"].max()
    final = df[df["epoch"] == final_epoch]
    smallest_domain_row = final.loc[final["domain_size"].idxmin()]

    return {
        "oversample_ratio_smallest_domain_final_epoch": float(
            smallest_domain_row["oversample_ratio"]
        ),
        "oversample_ratio_max_final_epoch": float(final["oversample_ratio"].max()),
    }


def load_coral_diag_summary(results_dir: str, tag: str) -> dict:
    path = os.path.join(results_dir, f"{tag}_coral_diag.csv")
    if not os.path.exists(path):
        return {
            "img_spatial_min_final_epoch": None,
            "roi_min_final_epoch": None,
            "roi_fg_min_final_epoch": None,
            "num_domains_present_mean_final_epoch": None,
        }
    df = pd.read_csv(path)
    if df.empty:
        return {
            "img_spatial_min_final_epoch": None,
            "roi_min_final_epoch": None,
            "roi_fg_min_final_epoch": None,
            "num_domains_present_mean_final_epoch": None,
        }

    final = df.iloc[df["epoch"].idxmax()]
    return {
        "img_spatial_min_final_epoch": float(final["img_spatial_min"]),
        "roi_min_final_epoch": float(final["roi_min"]),
        "roi_fg_min_final_epoch": float(final["roi_fg_min"]),
        "num_domains_present_mean_final_epoch": float(final["num_domains_present"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="coral_ablation_runs")
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Run tags, e.g. capped2_bs2 capped2_bs8 diverse_bs8 capped4_bs8 natural_bs8",
    )
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    rows = []
    for tag in args.tags:
        try:
            cfg = load_config(args.results_dir, tag)
        except FileNotFoundError:
            print(f"!! Missing config for {tag}, skipping")
            continue

        row = {"tag": tag, **cfg}
        row.update(load_map(args.results_dir, tag, "val"))
        row.update(load_map(args.results_dir, tag, "test"))
        row.update(load_domain_histogram_summary(args.results_dir, tag))
        row.update(load_coral_diag_summary(args.results_dir, tag))
        rows.append(row)

    if not rows:
        raise SystemExit("No runs found. Did run_coral_sampler_ablation.sh finish any points?")

    combined = pd.DataFrame(rows)

    display_cols = [
        "tag", "sampler", "batch_size", "K_domains_per_batch", "samples_per_domain",
        "val_map_50", "test_map_50",
        "oversample_ratio_smallest_domain_final_epoch",
        "img_spatial_min_final_epoch", "roi_min_final_epoch", "roi_fg_min_final_epoch",
    ]
    display_cols = [c for c in display_cols if c in combined.columns]

    print("\nCORAL sampling-ablation results")
    print("================================")
    print(combined[display_cols].to_string(index=False))

    print(
        "\nReading this table:\n"
        "- capped2_bs2 vs capped2_bs8: same sampler/K, samples_per_domain 1->4.\n"
        "  If test_map_50 differs a lot here but oversample_ratio_smallest_domain\n"
        "  barely moves, oversampling isn't the driver -- look at samples_per_domain\n"
        "  effects on img_spatial_min/roi_min/roi_fg_min instead.\n"
        "- capped2_bs8 vs diverse_bs8: same samples_per_domain-ish scale/steps-per-\n"
        "  epoch, K changes 2->8. If test_map_50 tracks K instead, the pairwise-CORAL\n"
        "  loss structure (mean_pairwise_coral over K domains) is the likelier culprit.\n"
        "- capped4_bs8 is the tiebreaker between the two stories above.\n"
        "- natural_bs8 shows what 'no domain control' looks like as a sanity check.\n"
        "- roi_fg_min tends to be the smallest number in the table by a wide margin --\n"
        "  if it's near/at 0 in the worse-performing configs, foreground ROI scarcity\n"
        "  (not oversampling) is the leading suspect for ds_adv/ds_cls specifically."
    )

    if args.out_csv:
        combined.to_csv(args.out_csv, index=False)
        print(f"\nSaved combined table to {args.out_csv}")


if __name__ == "__main__":
    main()
