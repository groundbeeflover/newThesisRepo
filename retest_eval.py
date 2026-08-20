#!/usr/bin/env python
"""re-runs a split's map eval off an already trained checkpoint and writes the
results into their own folder, leaving the original run csvs untouched

why this exists instead of just re-running the training script with --eval_map

  1. --eval_map writes {weights_file}_{split}_map.csv straight into
     --weights_folder, which is where the original run's csvs already are, so
     re-running in place would overwrite the exact numbers this is supposed to
     be compared against
  2. the per domain map@50 in the original runs has two problems and one
     definitional mismatch, and it's better to have the fixes in one file than
     patched separately into three training scripts that then drift apart

what's different from the evaluate_map in the training scripts

  a. the -1 sentinel, this is the fix from commit 60a066f. torchmetrics follows
     pycocotools and hands back map_50 = -1 when there's no ground truth to
     score against, which on a per image compute fires for every "no_box"
     image. the original averaged those in as if they were scores, which is
     what pulled test Terraref_2 negative and val UQ_2 down to about -0.6.
     carried over here so the coral runs get it too

  b. the missing reset. the original only called per_domain_metric.reset() in
     the "domain already seen" branch, so the first image of each domain was
     left sitting in the accumulator and leaked into the next update, which
     made the second image of every domain a 2 image pooled score instead of
     its own. reset is unconditional here

  c. map_50_pooled, a new column. seemakurthy et al define map@50 by pooling
     tp/fp/fn over the whole evaluation set and taking the area under a single
     precision recall curve (eq. 26-31), never by averaging per image scores.
     so the per image mean, fixed or not, is not the metric the paper reports.
     map_50_pooled accumulates all of a domain's images into one metric and
     computes once, which is that definition restricted to one domain, and is
     the same kind of number as the overall map_50 in the other csv

both per domain columns get written, so old and new can be diffed and neither
has to be taken on trust. there is deliberately no plain "map_50" column, the
two mean different things and a script silently picking up the wrong one is a
worse outcome than a KeyError

the overall map/map_50/map_75 numbers are computed exactly as before and should
reproduce the original csvs to within nondeterministic-inference noise

usage
  python retest_eval.py --kind dgkarthik \
      --weights_folder runs/gwhd_dgkarthik_nondet_lr1e-5_reg055/checkpoints \
      --weights_file dgkarthik_nondet_lr1e-5_reg055_round0 \
      --out_dir runs/gwhd_dgkarthik_nondet_lr1e-5_reg055/second_tests
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import pandas as pd
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

#kind -> (module to import, --exp value the checkpoint was trained with)
#exp matters because it decides which auxiliary heads the module builds, and
#those heads are in the state_dict, so getting it wrong is a load error
KINDS = {
    "baseline": ("train_GWHD_baseline_clean", None),
    "dgkarthik": ("train_GWHD_dgfrcnn_mattia", "dg"),
    "coral": ("train_GWHD_coralfrcnn", "coral"),
}

ANNOTS = {
    "val": "data/Annots/competition_val.csv",
    "test": "data/Annots/competition_test.csv",
}
#the training scripts pass image_set="tes" for test, kept verbatim so the
#dataset behaves identically, whatever it does with that string
IMAGE_SET = {"val": "val", "test": "tes"}


def build_module_args(module, kind: str, weights_folder: str, weights_file: str):
    """get a fully defaulted args namespace out of the module's own parser

    going through the real parser rather than hand rolling a namespace means
    every default (weight_decay, optimizer, num_domains, reg_weights, ...)
    matches what training used, without this file having to know them
    """
    argv = [
        "retest_eval",
        "--weights_folder", weights_folder,
        "--weights_file", weights_file,
    ]
    exp = KINDS[kind][1]
    if exp is not None:
        argv += ["--exp", exp]

    saved = sys.argv
    try:
        sys.argv = argv
        return module.parser_args()
    finally:
        sys.argv = saved


def build_detector(module, kind: str, args):
    """same constructor call the training script's __main__ makes"""
    if kind == "baseline":
        return module.CleanFasterRCNN(
            n_classes=2, lr=args.lr, weight_decay=args.weight_decay
        )
    if kind == "dgkarthik":
        return module.DGFRCNN(2, args.batch_size, args.exp, args.reg_weights, args)
    if kind == "coral":
        return module.DGFRCNN(
            2,
            args.batch_size,
            args.exp,
            args.reg_weights,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
        )
    raise ValueError(kind)


def as_boxes(raw, device):
    """[n,4] float box tensor on device, empty images included

    a "no_box" image comes through as an empty tensor and torchmetrics wants it
    shaped [0,4] rather than [0], so the reshape isn't optional
    """
    boxes = raw.float().to(device)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    return boxes.reshape(-1, 4)


def to_cpu_detached(entries):
    return [
        {k: v.detach().cpu() for k, v in entry.items() if torch.is_tensor(v)}
        for entry in entries
    ]


@torch.no_grad()
def evaluate(detector, dataloader, device):
    detector.eval()
    detector.to(device)

    domain_names = getattr(dataloader.dataset, "domain_names", {})

    #overall, one compute over the whole split, this is the headline number and
    #is computed the same way the original evaluate_map did
    overall = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        iou_thresholds=[0.1, 0.5, 0.75],
    ).to(device)

    #per image, reset after every single image, then averaged per domain over
    #the images that could actually be scored. this is the old metric with
    #fixes (a) and (b) applied
    per_image = MeanAveragePrecision(
        iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
    ).to(device)
    per_image_values: dict[int, list[float]] = {}

    #per domain pooled, one accumulator per domain, computed once at the end.
    #kept on cpu because 18 of these holding every detection would otherwise sit
    #in gpu memory next to the detector for the whole loop
    pooled: dict[int, MeanAveragePrecision] = {}
    domain_image_counts: dict[int, int] = {}

    for batch in tqdm(dataloader, desc="mAP evaluation"):
        images = [image.to(device) for image in batch[0]]

        targets = []
        for raw in batch[1]:
            boxes = as_boxes(raw, device)
            targets.append({
                "boxes": boxes,
                "labels": torch.ones(
                    boxes.shape[0], dtype=torch.long, device=device
                ),
            })

        predictions = detector(images)

        overall.update(predictions, targets)

        domain = int(batch[2][0].item())
        domain_image_counts[domain] = domain_image_counts.get(domain, 0) + 1

        #(b) unconditional reset, every image is scored on its own
        per_image.reset()
        per_image.update(predictions, targets)
        per_image_values.setdefault(domain, []).append(
            float(per_image.compute()["map_50"].item())
        )

        if domain not in pooled:
            pooled[domain] = MeanAveragePrecision(
                iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
            )
        pooled[domain].update(to_cpu_detached(predictions), to_cpu_detached(targets))

    results = overall.compute()

    print("\nEvaluation results")
    print("==================")
    print(f"mAP:     {results['map'].item():.6f}")
    print(f"mAP@50:  {results['map_50'].item():.6f}")
    print(f"mAP@75:  {results['map_75'].item():.6f}")

    print("\nPer-domain mAP@50")
    print("==================")
    print(f"{'domain':>14}  {'pooled':>9}  {'img_mean':>9}  {'scored':>7}  {'imgs':>5}")

    rows = []
    for domain in sorted(pooled.keys()):
        #(a) -1 is "could not be scored", not a score
        values = per_image_values[domain]
        scored = [v for v in values if v >= 0]
        image_mean = sum(scored) / len(scored) if scored else float("nan")

        #(c) the paper's definition, restricted to this domain
        pooled_map_50 = float(pooled[domain].compute()["map_50"].item())
        #a domain with no ground truth anywhere still comes back as -1 here,
        #and that is genuinely unscorable rather than a number to average
        if pooled_map_50 < 0:
            pooled_map_50 = float("nan")

        name = domain_names.get(domain, str(domain))
        n_images = domain_image_counts[domain]
        print(
            f"{name:>14}  {pooled_map_50:>9.4f}  {image_mean:>9.4f}  "
            f"{len(scored):>7}  {n_images:>5}"
        )

        rows.append({
            "domain_index": domain,
            "domain_name": name,
            "num_images": n_images,
            "num_images_scored": len(scored),
            "map_50_pooled": pooled_map_50,
            "map_50_image_mean": image_mean,
        })

    return (
        {
            "map_mean_iou_10_50_75": float(results["map"].item()),
            "map_50": float(results["map_50"].item()),
            "map_75": float(results["map_75"].item()),
        },
        rows,
    )


def main():
    parser = argparse.ArgumentParser(
        description="re-run a split's map eval from an existing checkpoint"
    )
    parser.add_argument("--kind", required=True, choices=sorted(KINDS))
    parser.add_argument(
        "--weights_folder", required=True,
        help="folder holding <weights_file>.ckpt, normally runs/<run>/checkpoints",
    )
    parser.add_argument("--weights_file", required=True, help="checkpoint stem, no .ckpt")
    parser.add_argument(
        "--out_dir", required=True,
        help="where the csvs go, kept separate from --weights_folder on purpose",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--data_root", default="data/gwhd_2021/images/",
        help="passed straight to WheatDataset as root_dir",
    )
    args = parser.parse_args()

    module_name = KINDS[args.kind][0]
    module = importlib.import_module(module_name)

    ckpt_path = os.path.join(args.weights_folder, args.weights_file + ".ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    module_args = build_module_args(
        module, args.kind, args.weights_folder, args.weights_file
    )

    detector = build_detector(module, args.kind, module_args)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = detector.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if missing or unexpected:
        #strict=False so a projection buffer that moved doesn't kill the run,
        #but anything in the backbone or the heads showing up here means the
        #wrong --kind or --exp and the numbers below would be meaningless
        print(f"!! missing keys ({len(missing)}): {missing[:10]}")
        print(f"!! unexpected keys ({len(unexpected)}): {unexpected[:10]}")
        if any(not k.startswith("coral_losses.") for k in missing + unexpected):
            raise RuntimeError(
                "state_dict mismatch outside the coral projection buffers, "
                f"check --kind (got {args.kind})"
            )

    dataset = module.WheatDataset(
        ANNOTS[args.split],
        root_dir=args.data_root,
        image_set=IMAGE_SET[args.split],
        transform=module.valid_transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=module.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating {args.split} split on {device}...")

    results, per_domain_rows = evaluate(detector, dataloader, device)

    os.makedirs(args.out_dir, exist_ok=True)

    overall_path = os.path.join(
        args.out_dir, f"{args.weights_file}_{args.split}_map.csv"
    )
    pd.DataFrame([{"split": args.split, **results}]).to_csv(overall_path, index=False)
    print(f"\nSaved results to {overall_path}")

    per_domain_path = os.path.join(
        args.out_dir, f"{args.weights_file}_{args.split}_per_domain_map.csv"
    )
    pd.DataFrame(
        [{"split": args.split, **row} for row in per_domain_rows]
    ).to_csv(per_domain_path, index=False)
    print(f"Saved per-domain results to {per_domain_path}")


if __name__ == "__main__":
    main()
