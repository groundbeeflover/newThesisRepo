from __future__ import absolute_import, division, print_function

"""
plain faster r-cnn baseline for gwhd, straight off torchvision's
fasterrcnn_resnet50_fpn instead of the repo's own fasterrcnn.py (fastwilds)
wrapper that the train_GWHD_dgfrcnn* scripts use for both dg and non_dg

why this exists: going off petra and mattia's email (2026-08-12, "dgod code"),
mattia's reproduction of seemakurthy et al did not use karthik's non_dg path for
the baseline number, he trained a fresh unmodified torchvision faster r-cnn.
the custom fasterrcnn.py rewrites the rpn/roi head losses to be per image rather
than per batch, which the da heads need for bookkeeping but is still a real
behavioural difference from stock torchvision even with those heads off. i
reckon that's why the earlier baseline vs grl comparisons here
(grlbaselineruns/, grlbaseline1e-4/) came out tied instead of showing the gap
from the paper

everything other than the detector itself, so data pipeline, optimizer, lr
schedule, early stopping and determinism, is kept identical to
train_GWHD_dgfrcnn_mattia.py so the baseline and dgkarthik runs stay as
comparable as they can be
"""

import os
import sys
import random
import argparse
import pickle

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics.detection import MeanAveragePrecision

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

SEED = -1


# ---------------------------------------------------------------------------
#dataset, transforms and collate_fn copied straight out of
#train_GWHD_dgfrcnn_mattia.py so the data pipeline matches the dgkarthik run
#exactly, same csvs, same [0,1] scaled images, same box decoding and clipping
# ---------------------------------------------------------------------------

class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, csv_file, root_dir, image_set, transform=None):
        annotations = pd.read_csv(csv_file)
        self.image_set = image_set
        self.image_path = root_dir + annotations["image_name"]
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        unique_values = annotations["domain"].unique()
        unique_indices = {value: index for index, value in enumerate(unique_values)}
        print(unique_indices)
        self.domain_index = annotations["domain_index"] = annotations["domain"].map(
            unique_indices
        )
        #flip of unique_indices, lets evaluate_map turn a domain_index back
        #into the actual domain name, built separately per split so it only
        #lines up with indices that came out of this same dataset object
        self.domain_names = {index: value for value, index in unique_indices.items()}
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        imgp = self.image_path[idx]
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        domain = torch.tensor(self.domain_index[idx])

        if self.transform:
            bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] = (
                bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] - 1
            )
            bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] = (
                bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] - 1
            )
            transformed = self.transform(
                image=image, bboxes=bboxes, class_labels=["wheat_head"] * len(bboxes)
            )
            image_tr = transformed["image"] / 255.0
            bboxes = transformed["bboxes"]

        if len(bboxes) > 0:
            bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
            bboxes = torch.zeros((0, 4))

        return image_tr, bboxes, domain, image

    def decodeString(self, BoxesString):
        if BoxesString == "no_box":
            return np.zeros((0, 4))
        else:
            try:
                boxes = np.array(
                    [
                        np.array([int(i) for i in box.split(" ")])
                        for box in BoxesString.split(";")
                    ]
                )
                return boxes.astype(np.int32).clip(min=0, max=1023)
            except Exception:
                print(BoxesString)
                print("Submission is not well formatted. empty boxes will be returned")
                return np.zeros((0, 4))


train_transform = A.Compose(
    [
        ToTensorV2(p=1.0),
    ],
    p=1.0,
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_labels"], min_area=20
    ),
)

valid_transform = A.Compose(
    [
        ToTensorV2(p=1.0),
    ],
    p=1.0,
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_labels"], min_area=20
    ),
)


def collate_fn(batch):
    images = list()
    targets = list()
    orig_img = list()
    domain_labels = list()
    for i, t, d, io in batch:
        images.append(i)
        targets.append(t)
        orig_img.append(io)
        domain_labels.append(d)
    images = torch.stack(images, dim=0)
    return images, targets, torch.tensor(domain_labels), orig_img


def mean_scored_map(values):
    """mean of one domain's per image map@50, skipping the unscorable images

    torchmetrics follows the pycocotools convention where map_50 comes back as
    -1 when a class has no ground truth instances at all, which for a per image
    compute fires on every image whose BoxesString is "no_box". that -1 is a
    "could not be scored" flag and not a score, so averaging it in drags the
    domain mean below zero, which is how gwhd val UQ_2 (13 no_box images out of
    16) ended up around -0.6 no matter how good the detector was. test
    Terraref_2 and CIMMYT_3 were being deflated the same way, just not far
    enough to go negative and get noticed

    comes back as (mean, num_scored), mean is nan if a domain had no scorable
    image at all
    """
    values = torch.stack(values)
    scored = values[values >= 0]
    if scored.numel() == 0:
        return float("nan"), 0
    return float(torch.mean(scored).item()), int(scored.numel())


# ---------------------------------------------------------------------------
#the model, stock torchvision faster r-cnn with none of the repo's changes
# ---------------------------------------------------------------------------

class CleanFasterRCNN(LightningModule):
    def __init__(self, n_classes, lr, weight_decay):
        super().__init__()

        #stock torchvision builder, weights=None plus the default
        #weights_backbone (imagenet resnet50) is the same as the
        #pretrained=False, pretrained_backbone=True used everywhere else in this
        #repo, so the backbone starts off comparable to the dgkarthik run and
        #only the detector itself is different
        #
        #image_mean=[0,0,0] and image_std=[1,1,1] because WheatDataset only
        #divides the pixels by 255, it never does the imagenet mean/std bit
        #leaving the real torchvision defaults in would normalize twice without
        #telling you, so this is the same trick fastwilds pulls in fasterrcnn.py
        self.detector = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=n_classes,
            min_size=1024,
            max_size=1024,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        )

        self.base_lr = lr
        self.weight_decay = weight_decay

        self.metric = MeanAveragePrecision(
            iou_type="bbox",
            class_metrics=True,
            iou_thresholds=[0.1, 0.5, 0.75],
            extended_summary=True,
        )
        self.per_domain_metric = MeanAveragePrecision(
            iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
        )
        self.per_domain_mAP = {}
        self.pr_file = "baseline"

    def forward(self, imgs, targets=None):
        #plain torchvision generalizedrcnn, gives back a loss dict in
        #training and a list of {boxes, labels, scores} in eval, no per image
        #loss splitting and no rpn/roi head overrides
        return self.detector(imgs, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.detector.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "max",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08,
            ),
            "monitor": "val_acc",
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        imgs = list(image.to(device=self.device) for image in batch[0])

        targets = []
        for boxes, domain in zip(batch[1], batch[2]):
            target = {}
            target["boxes"] = boxes.float().to(device=self.device)
            target["labels"] = (
                torch.ones(len(target["boxes"])).long().to(device=self.device)
            )
            targets.append(target)

        self.detector.train()
        loss_dict = self.detector(imgs, targets)
        loss = sum(loss_dict.values())

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(imgs))
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True, batch_size=len(imgs))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        img, boxes, labels, domain = batch

        self.detector.eval()
        with torch.no_grad():
            preds = self.detector(list(im.to(device=self.device) for im in img))

        targets = []
        for boxes, domain in zip(batch[1], batch[2]):
            target = {}
            target["boxes"] = boxes.float().to(device=self.device)
            target["labels"] = (
                torch.ones(len(target["boxes"])).long().to(device=self.device)
            )
            targets.append(target)

        try:
            self.metric.update(preds, targets)
            self.per_domain_metric.update(preds, targets)

            domain = domain.item()
            if domain in self.per_domain_mAP.keys():
                self.per_domain_mAP[domain].append(
                    self.per_domain_metric.compute()["map_50"].detach().cpu()
                )
                self.per_domain_metric.reset()
            else:
                self.per_domain_mAP[domain] = [
                    self.per_domain_metric.compute()["map_50"].detach().cpu()
                ]
        except Exception:
            print(targets)

    def on_validation_epoch_end(self):
        metrics = self.metric.compute()

        self.log("val_acc", metrics["map_50"])
        print(metrics["map_per_class"], metrics["map_50"])
        self.metric.reset()

        os.makedirs("helpers", exist_ok=True)
        with open("helpers/" + self.pr_file + ".pkl", "wb") as f:
            pickle.dump(metrics["precision"].cpu(), f)

        for key in self.per_domain_mAP.keys():
            mean_map_50, num_scored = mean_scored_map(self.per_domain_mAP[key])
            print(
                key,
                mean_map_50,
                num_scored,
                len(self.per_domain_mAP[key]),
            )


def evaluate_map(detector, dataloader, device):
    """global map over one whole split, nothing to do with the lightning loop

    gives the full map/map_50/map_75 breakdown and is what --eval_map runs after
    training so the results go into a per run csv, same schema as
    train_GWHD_coralfrcnn.py's --eval_map path so either script's output works
    with summarize_eval_results.py

    also does a per domain map@50 breakdown, printed here and handed back as a
    second value, the per domain tracking in the lightning
    validation_step/on_validation_epoch_end piles up across every validation
    epoch of a whole run so it isn't a clean single split snapshot, whereas the
    dict in here is local to one call and is exactly one pass over the dataloader

    comes back as (results, per_domain_rows), results being the usual
    map/map_50/map_75 dict and per_domain_rows a list of
    {"domain_index", "domain_name", "num_images", "num_images_scored",
    "map_50"}, one per domain in the dataloader, domain_name comes off
    dataloader.dataset.domain_names and falls back to the raw index as a string
    if that isn't there

    num_images is every image in the domain and num_images_scored is how many
    of them torchmetrics could actually score, the two differ on domains with
    "no_box" images and map_50 is the mean over the scored ones only, see
    mean_scored_map
    """
    detector.eval()
    detector.to(device)

    domain_names = getattr(dataloader.dataset, "domain_names", {})

    metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        iou_thresholds=[0.1, 0.5, 0.75],
        extended_summary=True,
    ).to(device)

    per_domain_metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        iou_thresholds=[0.5],
    ).to(device)
    per_domain_mAP = {}

    for batch in tqdm(dataloader, desc="mAP evaluation"):
        images = [image.to(device) for image in batch[0]]

        targets = []
        for boxes in batch[1]:
            boxes = boxes.float().to(device)
            targets.append({
                "boxes": boxes,
                "labels": torch.ones(len(boxes), dtype=torch.long, device=device),
            })

        with torch.no_grad():
            predictions = detector(images)

        metric.update(predictions, targets)

        #per domain breakdown, assumes batch_size=1 which is what the val
        #and test dataloaders here use anyway
        try:
            per_domain_metric.update(predictions, targets)
            domain = batch[2][0].item()
            if domain in per_domain_mAP:
                per_domain_mAP[domain].append(
                    per_domain_metric.compute()["map_50"].detach().cpu()
                )
                per_domain_metric.reset()
            else:
                per_domain_mAP[domain] = [
                    per_domain_metric.compute()["map_50"].detach().cpu()
                ]
        except Exception:
            print(targets)

    results = metric.compute()

    print("\nEvaluation results")
    print("==================")
    print(f"mAP:     {results['map'].item():.6f}")
    print(f"mAP@50:  {results['map_50'].item():.6f}")
    print(f"mAP@75:  {results['map_75'].item():.6f}")

    if results["map_per_class"].numel() > 0:
        print(f"AP per class: {results['map_per_class'].detach().cpu().tolist()}")

    print("\nPer-domain mAP@50 (domain, mean, num_scored, num_images)")
    print("==================")
    per_domain_rows = []
    for key in sorted(per_domain_mAP.keys()):
        mean_map_50, num_scored = mean_scored_map(per_domain_mAP[key])
        num_images = len(per_domain_mAP[key])
        print(key, mean_map_50, num_scored, num_images)
        per_domain_rows.append({
            "domain_index": key,
            "domain_name": domain_names.get(key, str(key)),
            "num_images": num_images,
            "num_images_scored": num_scored,
            "map_50": mean_map_50,
        })

    return {
        "map_mean_iou_10_50_75": float(results["map"].item()),
        "map_50": float(results["map_50"].item()),
        "map_75": float(results["map_75"].item()),
    }, per_domain_rows


def parser_args():
    parser = argparse.ArgumentParser(description="Clean torchvision Faster R-CNN baseline")

    parser.add_argument(
        "--weights_folder", dest="weights_folder", default="GWHD", type=str,
        help="Directory checkpoints are written to / read from.",
    )
    parser.add_argument(
        "--weights_file", dest="weights_file", default="baseline_clean", type=str,
        help="Checkpoint filename without .ckpt.",
    )
    parser.add_argument("--lr", dest="lr", default=1e-5, type=float, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", dest="weight_decay", default=0.0005, type=float,
        help="AdamW weight decay (matches train_GWHD_dgfrcnn_mattia.py's hardcoded value).",
    )
    parser.add_argument("--batch_size", dest="batch_size", default=8, type=int)
    parser.add_argument("--num_workers", dest="num_workers", default=16, type=int)
    parser.add_argument("--max_epochs", dest="max_epochs", default=100, type=int)
    parser.add_argument(
        "--accumulate_grad_batches", dest="accumulate_grad_batches", default=1, type=int,
        help="Lightning gradient accumulation steps. Use this to hit a target *effective* "
        "batch size (--batch_size * this value) on a GPU too small to fit it in one physical "
        "step -- e.g. --batch_size 4 --accumulate_grad_batches 2 for an effective BS=8 on a "
        "24GB card. Default 1 = no accumulation, matches --batch_size exactly (unchanged "
        "behavior). Note this is not bit-identical to a true single-step BS=8 forward pass: "
        "any non-frozen BatchNorm layers in the backbone compute statistics per physical step, "
        "not per effective batch.",
    )
    parser.add_argument(
        "--seed", dest="seed", default=42, type=int,
        help="Seed value for removing randomicity.",
    )
    parser.add_argument(
        "--deterministic", dest="deterministic", action="store_true",
        help="Enable full determinism (cudnn deterministic algorithms, "
        "Trainer(deterministic=True)). Default is nondeterministic/benchmark mode.",
    )
    parser.add_argument(
        "--eval_map", dest="eval_map", action="store_true",
        help="Skip training entirely; load the checkpoint at "
        "--weights_folder/--weights_file.ckpt and evaluate mAP (map/map_50/map_75) "
        "on --eval_split, writing <weights_file>_<split>_map.csv into --weights_folder. "
        "Matches train_GWHD_coralfrcnn.py's --eval_map convention.",
    )
    parser.add_argument(
        "--eval_split", dest="eval_split", default="test", choices=["val", "test"],
        help="Which split --eval_map evaluates. Default test (primary thesis metric).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()

    SEED = args.seed
    print(f"SEED: {SEED}")

    #determinism block, copied verbatim out of mattia dutto's email
    #(2026-08-06, forwarded by petra bosilj 2026-08-12), seeds
    #random/numpy/lightning/torch and flips the cudnn determinism switches
    random.seed(SEED)
    np.random.seed(SEED)
    pytorch_lightning.seed_everything(SEED)
    seed_everything(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    NET_FOLDER = args.weights_folder
    weights_file = args.weights_file

    tr_dataset = WheatDataset(
        "data/Annots/competition_train.csv",
        root_dir="data/gwhd_2021/images/",
        image_set="train",
        transform=train_transform,
    )
    vl_dataset = WheatDataset(
        "data/Annots/competition_val.csv",
        root_dir="data/gwhd_2021/images/",
        image_set="val",
        transform=valid_transform,
    )
    test_dataset = WheatDataset(
        "data/Annots/competition_test.csv",
        root_dir="data/gwhd_2021/images/",
        image_set="test",
        transform=valid_transform,
    )

    train_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        vl_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    detector = CleanFasterRCNN(n_classes=2, lr=args.lr, weight_decay=args.weight_decay)

    #--eval_map is a separate run after training (see run_repro_baseline.sh)
    #that loads the checkpoint this seed already trained and writes the results
    #csv, keeping it out of the training path means each seed gets its own train
    #log and its own test log instead of one merged file
    if args.eval_map:
        ckpt_path = os.path.join(NET_FOLDER, weights_file + ".ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        detector.load_state_dict(checkpoint["state_dict"])

        if args.eval_split == "val":
            eval_dataloader = val_dataloader
            split_name = "validation"
        else:
            eval_dataloader = test_dataloader
            split_name = "test"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\nEvaluating {split_name} split...")
        results, per_domain_rows = evaluate_map(detector=detector, dataloader=eval_dataloader, device=device)

        output_path = os.path.join(NET_FOLDER, f"{weights_file}_{args.eval_split}_map.csv")
        pd.DataFrame([{"split": args.eval_split, **results}]).to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")

        per_domain_output_path = os.path.join(
            NET_FOLDER, f"{weights_file}_{args.eval_split}_per_domain_map.csv"
        )
        pd.DataFrame([
            {"split": args.eval_split, **row} for row in per_domain_rows
        ]).to_csv(per_domain_output_path, index=False)
        print(f"Saved per-domain results to {per_domain_output_path}")

        sys.exit(0)

    if not os.path.exists(NET_FOLDER):
        os.makedirs(NET_FOLDER, exist_ok=True)

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=10, verbose=False, mode="max"
    )
    #save_last=True drops a NET_FOLDER/last.ckpt every epoch on top of the
    #best val_acc one under weights_file, that's what lets a run pick back up
    #across several slurm submissions when the partition's walltime cap
    #(education is 2h) is shorter than a full training run
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", dirpath=NET_FOLDER, filename=weights_file, mode="max",
        save_last=True,
    )
    #dirpath (NET_FOLDER) is shared by every seed of a repro run, run_repro_baseline.sh
    #passes the same --weights_folder each time and only changes --weights_file
    #with save_last=True that'd mean one generic last.ckpt sitting in a shared
    #folder, and the auto-resume check below would happily pick up the previous
    #seed's checkpoint instead of starting clean, so name the last checkpoint
    #after weights_file and keep each seed's resume state to itself
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"{weights_file}-last"

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=args.max_epochs,
        deterministic=args.deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=2,
    )

    #if this is a resubmission of a run the walltime cut off, pick up from
    #the last checkpoint, that restores the whole trainer state, weights,
    #optimizer, lr scheduler, epoch count and the early stopping patience
    last_ckpt_path = os.path.join(NET_FOLDER, f"{weights_file}-last.ckpt")
    resume_from = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    else:
        print("No existing checkpoint found, starting from scratch.")

    trainer.fit(
        detector,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_from,
    )

    #training stops here, no validate/test tacked on the end, run this script
    #again with --eval_map (see run_repro_baseline.sh) to score the checkpoint it just
    #wrote and get the results csv, which keeps each seed's test log and csv
    #separate from its training log, the .done sentinel gets written by the
    #shell wrapper once both halves finish, not here, since done now means
    #train and eval both went through
