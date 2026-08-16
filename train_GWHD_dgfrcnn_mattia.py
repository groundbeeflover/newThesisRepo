from __future__ import absolute_import, division, print_function

import math, sys, time, random, os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.autograd import Variable, Function

import pickle
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert
from torchmetrics.detection import MeanAveragePrecision

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning

# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# # seed_everything(SEED)

SEED = -1


class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, csv_file, root_dir, image_set, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        annotations = pd.read_csv(csv_file)
        self.image_set = image_set
        self.image_path = root_dir + annotations["image_name"]
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        # self.domains_str = annotations['domain']
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
        image = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # Opencv open images in BGR mode by default

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
            # bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] = bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] - 1
            # bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] = bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] - 1
            bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
            bboxes = torch.zeros((0, 4))

        return image_tr, bboxes, domain, image

    def decodeString(self, BoxesString):
        """
        Small method to decode the BoxesString
        """
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
            except:
                print(BoxesString)
                print("Submission is not well formatted. empty boxes will be returned")
                return np.zeros((0, 4))


# SIZE = 512
# This is for analysing the influence of augmentation on the performance
# All the individual augmentations are commented so as to get the true impact
# of proposed regularisation terms
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
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    """

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


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)


class _InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(_InstanceDA, self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        # self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        # self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer = nn.Linear(256, self.num_domains)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.dc_relu1(self.dc_ip1(x))
        x = self.dc_ip2(x)
        x = torch.sigmoid(self.classifer(x))

        return x


class _InsClsPrime(nn.Module):
    def __init__(self, num_cls):
        super(_InsClsPrime, self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        # self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        # self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer = nn.Linear(256, self.num_cls)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.dc_relu1(self.dc_ip1(x))
        x = self.dc_ip2(x)
        x = torch.sigmoid(self.classifer(x))

        return x


class _InsCls(nn.Module):
    def __init__(self, num_cls):
        super(_InsCls, self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        # self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        # self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer = nn.Linear(256, self.num_cls)

    def forward(self, x):
        x = self.dc_relu1(self.dc_ip1(x))
        x = self.dc_ip2(x)
        x = torch.sigmoid(self.classifer(x))

        return x


class _ImageDA(nn.Module):
    def __init__(self, dim, num_domains):
        super(_ImageDA, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(self.dim, 256, kernel_size=3, stride=4)
        self.Conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, self.num_domains)
        self.reLu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.reLu(self.Conv2(x))
        x = self.reLu(self.Conv3(x))
        x = self.reLu(self.Conv4(x))
        x = self.flatten(x)
        x = self.reLu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x


import fasterrcnn


class DGFRCNN(LightningModule):
    def __init__(self, n_classes, batchsize, exp, reg_weights, args):
        super(DGFRCNN, self).__init__()

        self.detector = fasterrcnn.fasterrcnn_resnet50_fpn(
            min_size=1024, max_size=1024, pretrained_backbone=True
        )
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, n_classes
        )
        self.n_classes = n_classes
        self.batchsize = batchsize
        self.exp = exp
        self.reg_weights = reg_weights
        self.num_domains = 18

        self.ImageDA = _ImageDA(256, self.num_domains)
        self.InsDA = _InstanceDA(self.num_domains)
        self.InsCls = nn.ModuleList(
            [_InsCls(self.n_classes) for i in range(self.num_domains)]
        )
        self.InsClsPrime = nn.ModuleList(
            [_InsClsPrime(self.n_classes) for i in range(self.num_domains)]
        )

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
        self.base_lr = args.lr  # Original base lr is 1e-4
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # self.automatic_optimization = False

        # Tapping the backbone features and region proposal features and its labels
        self.detector.backbone.register_forward_hook(self.store_backbone_out)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)

        self.mode = 0

    def store_ins_features(self, module, input1, output):
        self.box_features = output
        self.box_labels = input1[1]  # Torch tensor of size 512

    def store_backbone_out(self, module, input1, output):
        self.base_feat = output

    def forward(self, imgs, targets=None):
        # Torchvision FasterRCNN returns the loss during training
        # and the boxes during eval
        self.detector.eval()
        return self.detector(imgs)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.detector.parameters(),
                    "lr": self.base_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.ImageDA.parameters(),
                    "lr": self.base_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.InsDA.parameters(),
                    "lr": self.base_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.InsCls.parameters(),
                    "lr": self.base_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.InsClsPrime.parameters(),
                    "lr": self.base_lr,
                    "weight_decay": self.weight_decay,
                },
            ],
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
        # imgs = list(image for image in batch[0])

        targets = []
        for boxes, domain in zip(batch[1], batch[2]):
            target = {}
            target["boxes"] = boxes.float().to(device=self.device)
            target["labels"] = (
                torch.ones(len(target["boxes"])).long().to(device=self.device)
            )
            targets.append(target)

        # fasterrcnn takes both images and targets for training, returns
        # Detection using source images

        if self.mode == 0:

            loss_dict = {}
            # print(f"IMG_TYPE: {type(imgs)} - TRG_TYPE: {type(targets)}")
            # print(f"IMG_LEN: {len(imgs)} - TRG_LEN: {len(targets)}")
            # print(f"IMG_0_SHAPE: {imgs[0].shape} - TRG_0_LEN: {len(targets[0])}")
            # print(f"TRG_0_BOXES: {targets[0]['boxes'].shape} - TRG_0_LABELS: {targets[0]['labels'].shape}")
            # print(f"TARGET: {targets[0]}")
            # for i in range(len(targets)):
            #   if (targets[i]["boxes"].shape)[0] == (targets[i]['labels'].shape)[0]:
            #     print(f"{i}-element of the batch is okey - {(targets[i]['boxes'].shape)[0]}")
            #   else:
            #     print(f"{i}-element is not okey - {(targets[i]["boxes"].shape)[0]} - {(targets[i]['labels'].shape)[0]}")
            detections = self.detector(imgs, targets)

            # for n, p in self.detector.named_parameters():
            #   if p.grad is not None and not torch.isfinite(p.grad).all():
            #       print("bad grad in", n)
            #       break

            loss = sum(
                [
                    loss
                    for detection in detections
                    for loss in detection["losses"].values()
                ]
            )

            loss_dict["detection_loss"] = loss

            if self.exp == "dg":
                ImgDA_scores = self.ImageDA(self.base_feat["0"])
                loss_dict["DA_img_loss"] = self.reg_weights[0] * F.cross_entropy(
                    ImgDA_scores, batch[2].to(device=self.device)
                )
                IDA_out = self.InsDA(self.box_features)
                rep_factor = int(IDA_out.shape[0] / self.batchsize)
                ins_labels = (
                    batch[2]
                    .reshape(self.batchsize, 1)
                    .repeat(1, rep_factor)
                    .reshape(IDA_out.shape[0])
                )
                loss_dict["DA_ins_loss"] = self.reg_weights[1] * F.cross_entropy(
                    IDA_out, ins_labels.to(device=self.device)
                )
                ExpImgDA_scores = ImgDA_scores.repeat(1, rep_factor).reshape(
                    IDA_out.shape[0], self.num_domains
                )
                loss_dict["Cst_loss"] = self.reg_weights[2] * F.mse_loss(
                    IDA_out, ExpImgDA_scores
                )
                self.mode = 1

            loss = sum(loss1 for loss1 in loss_dict.values())

        elif (
            self.mode == 1
        ):  # Without recording the gradients for detector, we need to update the weights for classifier weights
            loss_dict = {}
            loss = []

            for index in range(len(self.InsCls)):
                for param in self.InsCls[index].parameters():
                    param.requires_grad = True

            for index in range(len(imgs)):
                with torch.no_grad():
                    _ = self.detector([imgs[index]], [targets[index]])
                cls_scores = self.InsCls[batch[2][index].item()](self.box_features)
                loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))

            loss_dict["cls"] = self.reg_weights[4] * (torch.mean(torch.stack(loss)))
            loss = sum(loss for loss in loss_dict.values())

            self.mode = 2

        elif (
            self.mode == 2
        ):  # Only the GRL Classification should influence the updates but here we need to update the detector weights as well
            loss_dict = {}
            loss = []

            for index in range(len(imgs)):
                _ = self.detector([imgs[index]], [targets[index]])
                cls_scores = self.InsClsPrime[batch[2][index].item()](self.box_features)
                loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))

            loss_dict["cls_prime"] = self.reg_weights[3] * (
                torch.mean(torch.stack(loss))
            )
            loss = sum(loss for loss in loss_dict.values())

            self.mode = 3

        else:  # For Mode 3

            loss_dict = {}
            loss = []
            consis_loss = []

            for index in range(len(self.InsCls)):
                for param in self.InsCls[index].parameters():
                    param.requires_grad = False

            for index in range(len(imgs)):
                _ = self.detector([imgs[index]], [targets[index]])
                temp = []
                for i in range(len(self.InsCls)):
                    if i != batch[2][index].item():
                        cls_scores = self.InsCls[i](self.box_features)
                        temp.append(cls_scores)
                        loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))
                consis_loss.append(
                    torch.mean(
                        torch.abs(
                            torch.stack(temp, dim=0)
                            - torch.mean(torch.stack(temp, dim=0), dim=0)
                        )
                    )
                )

            loss_dict["cls"] = self.reg_weights[4] * (
                torch.mean(torch.stack(loss))
            )  # + torch.mean(torch.stack(consis_loss)))
            loss = sum(loss for loss in loss_dict.values())

            self.mode = 0

        # self.manual_backward(loss)

        # GRADIENT CLIPPING - essential!
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return {"loss": loss}  # , "log": torch.stack(temp_loss).detach().cpu()}

    def validation_step(self, batch, batch_idx):

        img, boxes, labels, domain = batch

        preds = self.forward(img)

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

        except:
            print(targets)

    def on_validation_epoch_end(self):

        metrics = self.metric.compute()

        self.log("val_acc", metrics["map_50"])
        print(metrics["map_per_class"], metrics["map_50"])
        self.metric.reset()

        with open("helpers/" + self.pr_file + ".pkl", "wb") as f:
            pickle.dump(metrics["precision"].cpu(), f)

        for key in self.per_domain_mAP.keys():
            print(
                key,
                torch.mean(torch.stack(self.per_domain_mAP[key])),
                len(self.per_domain_mAP[key]),
            )


def evaluate_map(detector, dataloader, device):
    """
    Evaluate global mAP on one complete dataset split, independent of the
    Lightning training/validation loop. Gives a full map/map_50/map_75
    breakdown and is used by the --eval_map post-training path so results
    can be written to a per-run CSV -- same convention (and same CSV
    schema) as train_GWHD_coralfrcnn.py's --eval_map path, so results from
    any of the three training scripts are combinable with
    summarize_eval_results.py.
    """
    detector.eval()
    detector.to(device)

    metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        iou_thresholds=[0.1, 0.5, 0.75],
        extended_summary=True,
    ).to(device)

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

    results = metric.compute()

    print("\nEvaluation results")
    print("==================")
    print(f"mAP:     {results['map'].item():.6f}")
    print(f"mAP@50:  {results['map_50'].item():.6f}")
    print(f"mAP@75:  {results['map_75'].item():.6f}")

    if results["map_per_class"].numel() > 0:
        print(f"AP per class: {results['map_per_class'].detach().cpu().tolist()}")

    return {
        "map_mean_iou_10_50_75": float(results["map"].item()),
        "map_50": float(results["map_50"].item()),
        "map_75": float(results["map_75"].item()),
    }


def parser_args():
    parser = argparse.ArgumentParser(description="DGFRCNN Main Experiments")
    parser.add_argument(
        "--exp", dest="exp", help="non_dg or dg", default="non_dg", type=str
    )

    parser.add_argument(
        "--weights_folder",
        dest="weights_folder",
        help="Name of the weights folder",
        default="GWHD",
        type=str,
    )

    parser.add_argument(
        "--weights_file",
        dest="weights_file",
        help="Name of the weights file",
        default="baseline",
        type=str,
    )

    parser.add_argument(
        "--reg_weights",
        nargs=5,
        metavar=("a", "b", "c", "d", "e"),
        dest="reg_weights",
        help="Regularisation constats",
        type=float,
    )

    parser.add_argument(
        "--model-checkpoint",
        dest="model_checkpoint",
        help="Name of the model file used for the training",
        type=str,
    )
    parser.add_argument(
        "--lr", dest="lr", help="Learning rate", default=1e-4, type=float
    )

    parser.add_argument(
        "--batch_size", dest="batch_size", default=8, type=int,
        help="Physical per-step batch size (was hardcoded to 8; now overridable). "
        "Combine with --accumulate_grad_batches to hit an effective batch size (e.g. "
        "--batch_size 4 --accumulate_grad_batches 2 for effective BS=8) if the full "
        "physical batch OOMs on a smaller GPU.",
    )

    parser.add_argument(
        "--accumulate_grad_batches", dest="accumulate_grad_batches", default=1, type=int,
        help="Lightning gradient accumulation steps. Default 1 = no accumulation, "
        "unchanged behavior. Not bit-identical to a true single-step forward pass at the "
        "effective batch size: any non-frozen BatchNorm layers compute statistics per "
        "physical step, not per effective batch.",
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        help="Seed value for removing randomicity",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
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
        torch.use_deterministic_algorithms(
            True
        )  # This should also include the two before, but I'm not sure.
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    NET_FOLDER = args.weights_folder

    weights_file = args.weights_file

    # Dataloader design based on input arguments
    # Training Dataset
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
        image_set="tes",
        transform=valid_transform,
    )

    BATCH_SIZE = args.batch_size

    train_dataloader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=16,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        vl_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # Instantiating the detector
    detector = DGFRCNN(
        2, BATCH_SIZE, args.exp, args.reg_weights, args
    )  # Num classes + 1 and batch_size

    # --eval_map: separate, post-training invocation (see run_repro_dgkarthik.sh)
    # that loads the checkpoint this seed already trained and writes a results
    # CSV -- kept out of the training path so each seed gets its own train log
    # (this process's stdout) and its own test log (the --eval_map process's
    # stdout) instead of one merged log file.
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
        results = evaluate_map(detector=detector, dataloader=eval_dataloader, device=device)

        output_path = os.path.join(NET_FOLDER, f"{weights_file}_{args.eval_split}_map.csv")
        pd.DataFrame([{"split": args.eval_split, **results}]).to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")

        sys.exit(0)

    # if os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt'):
    #   detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
    # else:
    #   if not os.path.exists(NET_FOLDER):
    #     os.mkdir(NET_FOLDER, 0o777)

    if not os.path.exists(NET_FOLDER):
        os.mkdir(NET_FOLDER, 0o777)

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
    # NOTE: dirpath (NET_FOLDER) is shared across all seeds of a repro run
    # (run_repro_dgkarthik.sh passes the same --weights_folder for every
    # seed, only --weights_file differs). save_last=True would otherwise
    # write a single generic "last.ckpt" into that shared directory, so a
    # later seed's auto-resume check below would find and resume from the
    # *previous* seed's last checkpoint instead of starting fresh. Namespace
    # the "last" checkpoint by weights_file to keep each seed's resume state
    # isolated.
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"{weights_file}-last"

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=100,
        deterministic=args.deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=2,
    )

    # Resume from the last checkpoint if this is a re-submission of a run that
    # got cut off by the walltime limit (full trainer state: weights, optimizer,
    # LR scheduler, epoch count, early-stopping patience counter, and -- for this
    # script -- the manual self.mode state machine, which lives on the LightningModule
    # and is captured by the checkpoint like any other module attribute... actually
    # it isn't (self.mode is a plain int, not a registered buffer), so a resumed run
    # restarts its mode-cycle at 0 rather than wherever it left off. That's a minor
    # inefficiency (repeats up to 3 extra sub-steps), not a correctness problem.
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

    # Training ends here -- no inline post-fit validate/test. Run this same
    # script again with --eval_map (see run_repro_dgkarthik.sh) to evaluate the
    # checkpoint just written and get a results CSV; that gives each seed its
    # own test log and CSV, separate from this training log. The .done
    # sentinel is written by the shell wrapper once both steps succeed, not
    # here, since "done" now means train+eval both completed.

    #  val_acc            0.5711795687675476 with batch size 4 - baseline-v11 - 38 epochs.
    #  val_acc             0.622123122215271 with batch size 8 - baseline-v12 - 25 epochs.
    #  val_acc          0.00018613977590575814 with batch size 1 - baseline-v13 - 10 epochs - lr = 1e-3.