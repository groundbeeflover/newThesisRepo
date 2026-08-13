from __future__ import absolute_import, division, print_function

"""
Clean/pure Faster R-CNN baseline for GWHD, using torchvision's stock
`fasterrcnn_resnet50_fpn` builder directly instead of the repo's custom
`fasterrcnn.py` (FastWILDS) wrapper that `train_GWHD_dgfrcnn*.py` use for
both the DG and non_dg experiments.

Why this script exists: per Petra/Mattia's email (2026-08-12, "DGOD Code"),
Mattia's successful reproduction of Seemakurthy et al. did NOT use Karthik's
non_dg path for the baseline number -- he trained a fresh, un-modified
torchvision Faster R-CNN instead. The custom `fasterrcnn.py` module
overrides the RPN/ROI-head loss computation to be per-image rather than
per-batch (needed for the DA heads' bookkeeping), which is a real behavioral
difference from stock torchvision even with the DA heads switched off, and
is suspected to be why earlier baseline-vs-GRL comparisons in this repo
(grlbaselineruns/, grlbaseline1e-4/) came out statistically tied instead of
showing the paper's gap.

Everything *except* the detector itself (data pipeline, optimizer choice,
LR schedule, early stopping, determinism handling) is kept identical to
train_GWHD_dgfrcnn_mattia.py so the baseline and DGKarthik runs are as
comparable as possible.
"""

import os
import random
import argparse
import pickle

import numpy as np
import pandas as pd
import cv2

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
# Dataset / transforms / collate_fn: copied verbatim from
# train_GWHD_dgfrcnn_mattia.py for 1:1 data-pipeline parity with the
# DGKarthik reproduction run (same CSVs, same [0,1]-scaled-only images,
# same box decoding/clipping).
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


# ---------------------------------------------------------------------------
# Model: stock torchvision Faster R-CNN, no repo customization.
# ---------------------------------------------------------------------------

class CleanFasterRCNN(LightningModule):
    def __init__(self, n_classes, lr, weight_decay):
        super().__init__()

        # Stock torchvision builder. `weights=None` + default `weights_backbone`
        # (ImageNet-pretrained ResNet50) matches the repo's own convention of
        # `pretrained=False, pretrained_backbone=True` used everywhere else in
        # this codebase (fasterrcnn.py's fasterrcnn_resnet50_fpn) -- so the
        # backbone initialization is comparable to the DGKarthik run, only the
        # detector implementation itself differs.
        #
        # image_mean=[0,0,0]/image_std=[1,1,1]: WheatDataset only scales pixels
        # to [0,1] (see __getitem__, `image_tr = transformed["image"] / 255.0`),
        # it does not apply ImageNet mean/std normalization. Passing the actual
        # torchvision defaults here would silently double-normalize; this
        # matches the "images are already normalized" trick used elsewhere in
        # the repo (fasterrcnn.py's FastWILDS).
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
        # Stock torchvision GeneralizedRCNN: returns loss dict in training
        # mode, list of {boxes, labels, scores} dicts in eval mode -- no
        # per-image loss splitting or custom RPN/ROI-head overrides.
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
            print(
                key,
                torch.mean(torch.stack(self.per_domain_mAP[key])),
                len(self.per_domain_mAP[key]),
            )


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
        "--seed", dest="seed", default=42, type=int,
        help="Seed value for removing randomicity.",
    )
    parser.add_argument(
        "--deterministic", dest="deterministic", action="store_true",
        help="Enable full determinism (cudnn deterministic algorithms, "
        "Trainer(deterministic=True)). Default is nondeterministic/benchmark mode.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()

    SEED = args.seed
    print(f"SEED: {SEED}")

    # Determinism block, verbatim from Mattia Dutto's email (2026-08-06,
    # forwarded by Petra Bosilj 2026-08-12): random/numpy/pytorch_lightning/
    # torch seeding plus cudnn determinism switches.
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

    if not os.path.exists(NET_FOLDER):
        os.makedirs(NET_FOLDER, exist_ok=True)

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=10, verbose=False, mode="max"
    )
    # save_last=True: writes NET_FOLDER/last.ckpt every epoch (in addition to the
    # best-val_acc checkpoint under `weights_file`). This is what lets a run resume
    # cleanly across multiple SLURM submissions when the partition's walltime cap
    # (e.g. education = 2h) is shorter than a full training run.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", dirpath=NET_FOLDER, filename=weights_file, mode="max",
        save_last=True,
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=args.max_epochs,
        deterministic=args.deterministic,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=2,
    )

    # Resume from the last checkpoint if this is a re-submission of a run that
    # got cut off by the walltime limit (full trainer state: weights, optimizer,
    # LR scheduler, epoch count, early-stopping patience counter).
    last_ckpt_path = os.path.join(NET_FOLDER, "last.ckpt")
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

    ckpt_path = os.path.join(NET_FOLDER, weights_file + ".ckpt")
    detector.load_state_dict(torch.load(ckpt_path)["state_dict"])

    eval_trainer = Trainer(accelerator="gpu", max_epochs=0, num_sanity_val_steps=-1)

    print("VALIDATION:")
    eval_trainer.validate(detector, dataloaders=val_dataloader, ckpt_path=ckpt_path)

    print("TEST:")
    eval_trainer.validate(detector, dataloaders=test_dataloader, ckpt_path=ckpt_path)

    # Sentinel file so the SLURM wrapper can tell "training+eval genuinely finished"
    # apart from "best checkpoint exists but the run was cut off mid-training and
    # needs resubmitting to resume".
    with open(os.path.join(NET_FOLDER, weights_file + ".done"), "w") as f:
        f.write("done\n")
