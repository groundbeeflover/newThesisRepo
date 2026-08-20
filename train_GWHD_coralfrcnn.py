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

import pytorch_lightning
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dg import (
    FRCNNCoralLosses,
    DomainDiverseBatchSampler,
    DomainCappedBatchSampler,
    NaturalDomainBatchSampler,
)

#--seed in __main__ overwrites this, and it does it before anything that
#depends on the seed gets built, so the coral projection, the batch sampler and
#the model all see the right value
SEED = 42

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
        self.image_path = root_dir+annotations["image_name"]
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        #self.domains_str = annotations['domain']
        unique_values = annotations['domain'].unique()
        unique_indices = {value: index for index, value in enumerate(unique_values)}
        print(unique_indices)
        self.domain_index = annotations['domain_index'] = annotations['domain'].map(unique_indices)
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
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Opencv open images in BGR mode by default
        
        domain = torch.tensor(self.domain_index[idx])
       
                   
        if self.transform:
          bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] = bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] - 1
          bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] = bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] - 1
          transformed = self.transform(image=image,bboxes=bboxes,class_labels=["wheat_head"]*len(bboxes)) 
          image_tr = transformed["image"]/255.0
          bboxes = transformed["bboxes"]
          
        
        
        if len(bboxes) > 0:
          #bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] = bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] - 1
          #bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] = bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] - 1
          bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
          bboxes = torch.zeros((0,4))
          
               
        return image_tr, bboxes, domain, image
              
    def decodeString(self,BoxesString):
      """
      Small method to decode the BoxesString
      """
      if BoxesString == "no_box":
          return np.zeros((0,4))
      else:
          try:
              boxes =  np.array([np.array([int(i) for i in box.split(" ")]) for box in BoxesString.split(";")])
              return boxes.astype(np.int32).clip(min=0, max=1023)
          except:
              print(BoxesString)
              print("Submission is not well formatted. empty boxes will be returned")
              return np.zeros((0,4))



#SIZE = 512
#This is for analysing the influence of augmentation on the performance
#All the individual augmentations are commented so as to get the true impact
#of proposed regularisation terms
train_transform = A.Compose(
        [
        ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20)
    )


valid_transform = A.Compose([
    ToTensorV2(p=1.0),
],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))



def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    """

    images = list()
    targets=list()
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

def detection_accuracy(src_boxes, pred_boxes, iou_threshold=0.5):
    """seemakurthy style gwhd detection accuracy for one image, tp / (tp + fp + fn)

    same logic as the old train_GWHD.py, match preds to gt on iou, count tp/fp/fn
    and deal with the empty gt and empty prediction cases on their own
    """

    #keep it all on cpu, simpler for eval
    src_boxes = src_boxes.detach().cpu()
    pred_boxes = pred_boxes.detach().cpu()

    total_gt = len(src_boxes)
    total_pred = len(pred_boxes)

    if total_gt > 0 and total_pred > 0:
        matcher = Matcher(
            iou_threshold,
            iou_threshold,
            allow_low_quality_matches=False
        )

        match_quality_matrix = box_iou(src_boxes, pred_boxes)
        results = matcher(match_quality_matrix)

        true_positive = torch.count_nonzero(results.unique() != -1)
        matched_elements = results[results > -1]

        false_positive = torch.count_nonzero(results == -1) + (
            len(matched_elements) - len(matched_elements.unique())
        )
        false_negative = total_gt - true_positive

        denom = true_positive + false_positive + false_negative
        if denom == 0:
            return torch.tensor(0.0)

        return true_positive.float() / denom.float()

    elif total_gt == 0:
        if total_pred > 0:
            return torch.tensor(0.0)
        else:
            return torch.tensor(1.0)

    else:
        #there were gt boxes but we predicted nothing
        return torch.tensor(0.0)

import fasterrcnn
class DGFRCNN(LightningModule):
    def __init__(self, n_classes, batchsize, exp, reg_weights, lr=0.001, weight_decay=0.0005, optimizer_name='adamw'):
        super(DGFRCNN, self).__init__()
        
        self.detector = fasterrcnn.fasterrcnn_resnet50_fpn(min_size=1024, max_size=1024, pretrained_backbone=True) 
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features 
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.n_classes = n_classes
        self.batchsize = batchsize
        self.exp = exp
        self.reg_weights = reg_weights
        self.num_domains = 18
	
        #coral swaps the four grl classifier heads for one statistical loss
        #module, the roi projection is frozen so the extra branch can't just
        #collapse to a zero projection and call it a day
        self.coral_losses = FRCNNCoralLosses(
            image_dim=256,
            roi_dim=1024,
            common_dim=256,
            max_spatial_samples_per_domain=2048,
            max_roi_samples_per_group=256,
            covariance_eps=1e-5,
            projection_seed=SEED,
        )

        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.1, 0.5, 0.75], extended_summary=True)
        self.per_domain_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.5])
        self.per_domain_mAP = {}
        self.pr_file = 'baseline'
        self.base_lr = lr
        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        #instrumentation for the sampling ablation, __main__ fills these in
        #once the batch_sampler and the weights folder/file are known, all
        #optional, None just means skip it, so non_dg and eval runs are fine
        self.coral_batch_sampler = None
        self.domain_hist_csv_path = None
        self.coral_diag_csv_path = None
        self._epoch_diag_records = []
        self._domain_hist_rows = []
        self._coral_diag_rows = []


        # Tapping the backbone features and region proposal features and its labels
        self.detector.backbone.register_forward_hook(self.store_backbone_out)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)
        
       
    def store_ins_features(self, module, input1, output):
      self.box_features = output
      self.box_labels = input1[1] #Torch tensor of size 512
      
            
    def store_backbone_out(self, module, input1, output):
      self.base_feat = output
      
    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training 
      # and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      #FRCNNCoralLosses has nothing to train, its only projection is a fixed
      #buffer, the coral gradients go into the detector instead
      params = [
          {
              'params': self.detector.parameters(),
              'lr': self.base_lr,
              'weight_decay': self.weight_decay,
          }
      ]

      if self.optimizer_name == 'adamw':
          return torch.optim.AdamW(params)
      return torch.optim.Adam(params)

#data collected by collate_fn, inside a batch there are image tensors, ground truth bboxes, domain labels per img, original images
    def training_step(self, batch, batch_idx):
      imgs = [image.to(self.device) for image in batch[0]] #moving image tensors to the gpu, if exists,
      image_domains = batch[2].to(self.device).long() #gather domain labels to gpu


#create a custom formatting of the gt bboxes and move to gpu
      targets = []
      for boxes in batch[1]:
        boxes = boxes.float().to(self.device) 
        targets.append({
            "boxes": boxes,
            "labels": torch.ones(
                len(boxes),
                dtype=torch.long,
                device=self.device,
            ),
        })


      #one forward pass gets us the detection losses, the backbone features,
      #the roi features and the roi labels that were already assigned
      detections = self.detector(imgs, targets)
      detection_loss = sum(
          loss
          #for each image in a batch there is a detection object
          for detection in detections
          #for each image there is a series of losses which are summed
          for loss in detection["losses"].values()
      )
      #summed across images in batch


#start loss dictionary, the regular frcnn is already added
      loss_dict = {
          "detection_loss": detection_loss,
      }

      if self.exp == "coral":
        coral = self.coral_losses(
            image_feature_map=self.base_feat["0"],
            roi_features=self.box_features,
            roi_labels_by_image=self.box_labels,
            image_domains=image_domains,
        )
        diag = coral.pop("diag")
        self._record_coral_diag(diag)

        # reg_weights order:
        # alpha1 image, alpha2 instance, alpha3 consistency,
        # alpha4 domain-specific adversarial, alpha5 domain-specific classification
        loss_dict.update({
            "coral_img": self.reg_weights[0] * coral["img"],
            "coral_ins": self.reg_weights[1] * coral["ins"],
            "coral_cst": self.reg_weights[2] * coral["cst"],
            "coral_ds_adv": self.reg_weights[3] * coral["ds_adv"],
            "coral_ds_cls": self.reg_weights[4] * coral["ds_cls"],
        })

        #logging the raw losses too, the grl paper's alphas were never going
        #to carry over to coral's loss magnitudes as is
        for name, value in coral.items():
          self.log(
              f"train/coral_raw_{name}",
              value,
              on_step=True,
              on_epoch=True,
              prog_bar=False,
              batch_size=len(imgs),
          )

      loss = sum(loss_dict.values())

      self.log(
          "train_loss",
          loss,
          on_step=True,
          on_epoch=True,
          prog_bar=False,
          batch_size=len(imgs),
      )

      for name, value in loss_dict.items():
        self.log(
            f"train/{name}",
            value,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(imgs),
        )

      return {"loss": loss}

    def _record_coral_diag(self, diag: dict) -> None:
        """park one step's pre-cap sample counts so on_train_epoch_end can
        average them up later"""
        img_counts = list(diag["image_spatial_counts"].values())
        roi_counts = list(diag["roi_counts"].values())
        fg_counts = list(diag["roi_fg_counts"].values())

        record = {
            "num_domains_present": diag["num_domains_present"],
            "img_spatial_min": min(img_counts) if img_counts else 0,
            "img_spatial_mean": (sum(img_counts) / len(img_counts)) if img_counts else 0.0,
            "roi_min": min(roi_counts) if roi_counts else 0,
            "roi_mean": (sum(roi_counts) / len(roi_counts)) if roi_counts else 0.0,
            "roi_fg_min": min(fg_counts) if fg_counts else 0,
            "roi_fg_mean": (sum(fg_counts) / len(fg_counts)) if fg_counts else 0.0,
        }
        self._epoch_diag_records.append(record)

        #also log these per step, on_epoch=True means lightning averages them
        #over the epoch as well, which doubles up with the csv below but is
        #handy if i'm just watching tensorboard
        for name, value in record.items():
            self.log(
                f"train/coral_diag_{name}",
                float(value),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )

    def on_train_epoch_start(self):
        self._epoch_diag_records = []

    def on_train_epoch_end(self):
        """writes two diagnostic csvs for the sampling ablation, a row per epoch

        the domain histogram one has images_drawn, oversample_ratio and
        times_recycled per domain, straight off the active batch_sampler, the
        coral diag one has the epoch mean of the pre-cap sample counts from
        _record_coral_diag, so whether the 2048/256 caps were actually biting
        and how thin the foreground rois got per domain

        both quietly do nothing if __main__ never wired up the sampler or the
        paths, which is the case for non_dg and eval only runs
        """
        epoch_idx = self.current_epoch

        if self.coral_batch_sampler is not None and self.domain_hist_csv_path is not None:
            if hasattr(self.coral_batch_sampler, "get_last_epoch_stats"):
                for row in self.coral_batch_sampler.get_last_epoch_stats():
                    row = dict(row)
                    row["epoch"] = epoch_idx
                    self._domain_hist_rows.append(row)

                hist_df = pd.DataFrame(self._domain_hist_rows)
                hist_df.to_csv(self.domain_hist_csv_path, index=False)

                oversample_ratios = [
                    r["oversample_ratio"] for r in self._domain_hist_rows
                    if r["epoch"] == epoch_idx
                ]
                if oversample_ratios:
                    self.log(
                        "epoch_domain_oversample_max",
                        float(max(oversample_ratios)),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )

        if self.coral_diag_csv_path is not None and self._epoch_diag_records:
            keys = self._epoch_diag_records[0].keys()
            means = {
                k: sum(r[k] for r in self._epoch_diag_records) / len(self._epoch_diag_records)
                for k in keys
            }
            means["epoch"] = epoch_idx
            self._coral_diag_rows.append(means)

            diag_df = pd.DataFrame(self._coral_diag_rows)
            diag_df.to_csv(self.coral_diag_csv_path, index=False)

    def validation_step(self, batch, batch_idx):
    
        images, boxes_list, domain_labels, _ = batch
    
        images = [
            image.to(self.device)
            for image in images
        ]
    
        preds = self.forward(images)
    
        targets = []
    
        for boxes in boxes_list:
            boxes = boxes.float().to(self.device)
    
            targets.append({
                "boxes": boxes,
                "labels": torch.ones(
                    len(boxes),
                    dtype=torch.long,
                    device=self.device,
                ),
            })
    
        self.metric.update(preds, targets)
              
    def on_validation_epoch_end(self):
    
        metrics = self.metric.compute()
    
        val_map50 = metrics["map_50"]
    
        self.log(
            "val_acc",
            val_map50,
            prog_bar=True,
        )
    
        print(
            f"Validation mAP@50: "
            f"{val_map50.item():.6f}"
        )
    
        self.metric.reset()    
   
def parser_args():
  parser = argparse.ArgumentParser(description='DGFRCNN Main Experiments')
  parser.add_argument('--exp', dest='exp',
                      choices=['non_dg', 'coral'],
                      help='Train baseline Faster R-CNN or CORAL-DG Faster R-CNN',
                      default='non_dg', type=str)
                      
  parser.add_argument('--weights_folder', dest='weights_folder',
                      help='Name of the weights folder',
                      default='GWHD', type=str)
                      
  parser.add_argument('--weights_file', dest='weights_file',
                      help='Name of the weights file',
                      default='baseline', type=str)
  
  parser.add_argument('--reg_weights', nargs=5, metavar=('a', 'b', 'c', 'd', 'e'),
                       dest='reg_weights',
                       help='CORAL weights: image instance consistency ds_adv ds_cls',
                       default=[1.0, 1.0, 1.0, 1.0, 1.0],
                       type=float)

  parser.add_argument('--batch_size', default=2, type=int,
                      help='Training batch size. Paper value for GWHD Faster R-CNN is 2.')

  parser.add_argument('--sampler', default='domain_diverse',
                      choices=['domain_diverse', 'domain_capped', 'natural'],
                      help="CORAL batch sampling strategy. 'domain_diverse' (original) "
                           "forces batch_size distinct domains per batch, so samples-per-domain "
                           "is pinned at 1 regardless of batch_size. 'domain_capped' caps the "
                           "number of distinct domains per batch at --domains_per_batch, so "
                           "samples-per-domain grows with batch_size instead. 'natural' uses "
                           "plain random shuffling (same as the non-DG baseline) with only a "
                           "minimum-2-domains-per-batch safeguard.")

  parser.add_argument('--domains_per_batch', default=2, type=int,
                      help="Number of distinct domains per batch when --sampler=domain_capped.")

  parser.add_argument('--lr', default=0.001, type=float,
                      help='Learning rate. Paper value for GWHD Faster R-CNN is 0.001.')

  parser.add_argument('--weight_decay', default=0.0005, type=float,
                      help='Weight decay. Paper value for GWHD Faster R-CNN is 0.0005.')

  parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'adam'],
                      help='Optimizer. Paper value for GWHD Faster R-CNN is AdamW.')

  parser.add_argument('--max_epochs', default=100, type=int,
                      help='Maximum number of epochs before early stopping.')

  parser.add_argument('--num_workers', default=16, type=int,
                      help='Number of dataloader workers.')

  parser.add_argument(
      '--accumulate_grad_batches', dest='accumulate_grad_batches', default=1, type=int,
      help='Lightning gradient accumulation steps. Use this to hit a target *effective* '
      'batch size (--batch_size * this value) on a GPU too small to fit it in one physical '
      'step -- e.g. --batch_size 4 --accumulate_grad_batches 2 for an effective BS=8 on a '
      '24GB card. Default 1 = no accumulation, matches --batch_size exactly (unchanged '
      'behavior). Note this is not bit-identical to a true single-step BS=8 forward pass: '
      'any non-frozen BatchNorm layers in the backbone compute statistics per physical step, '
      'not per effective batch, and (for --sampler domain_diverse) the number of distinct '
      'domains guaranteed per physical batch also shrinks to --batch_size.',
  )

  parser.add_argument(
      '--seed', dest='seed', default=42, type=int,
      help='Seed value for removing randomicity.',
  )

  parser.add_argument(
      '--deterministic', dest='deterministic', action='store_true',
      help='Enable full determinism (cudnn deterministic algorithms, '
      'Trainer(deterministic=True)). Default is nondeterministic/benchmark mode.',
  )

  parser.add_argument('--eval_wada', action='store_true',
                      help='Run WADA/ADA-style evaluation on the test set instead of training.')

  parser.add_argument('--score_threshold', default=0.5, type=float,
                      help='Prediction confidence threshold for WADA evaluation.')

  parser.add_argument('--iou_threshold', default=0.5, type=float,
                      help='IoU threshold for WADA matching.')
  parser.add_argument(
    "--eval_map",
    action="store_true",
    help="Evaluate mAP on the validation or test split and exit.",)
  
  parser.add_argument('--eval_split', default='test', choices=['val', 'test'],
                      help='Which split to evaluate with WADA.')
    
  parser.add_argument('--wada_output_dir', default=None, type=str,
                      help='Optional output directory for WADA results.')              
 
  parser.add_argument('--wada_max_batches', default=None, type=int,
                    help='Optional max number of batches/images for WADA smoke testing.')
  
  return parser.parse_args()

@torch.no_grad()
def evaluate_wada(detector, dataloader, output_dir, score_threshold=0.5, iou_threshold=0.5, max_batches=None):
    """runs a wada/ada style metric over a dataloader

    hands back the per domain scores, wada (plain mean of those) and
    global_accuracy (mean over every image, not domain balanced)
    """

    detector.eval()
    detector.to("cuda" if torch.cuda.is_available() else "cpu")
    device = detector.device if hasattr(detector, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_domain_scores = {}
    all_image_scores = []

    for batch_idx, data_sample in enumerate(tqdm(dataloader, desc="WADA evaluation")):
      if max_batches is not None and batch_idx >= max_batches:
              break
        
      images, boxes_list, domain_labels, orig_imgs = data_sample

        #wada eval wants a dataloader batch size of 1
      images = images.to(device)

      preds = detector(images)
      for item_idx, pred in enumerate(preds):
          gt_boxes = boxes_list[item_idx]
          pred_boxes = pred["boxes"].detach().cpu()
          pred_scores = pred["scores"].detach().cpu()
          keep = pred_scores > score_threshold
          pred_boxes = pred_boxes[keep]
          domain = int(domain_labels[item_idx].item())
          image_score = detection_accuracy(
              gt_boxes,
              pred_boxes,
              iou_threshold=iou_threshold
          ).detach().cpu()
          if domain not in per_domain_scores:
              per_domain_scores[domain] = []
          per_domain_scores[domain].append(image_score)
          all_image_scores.append(image_score)

    domain_rows = []
    for domain in sorted(per_domain_scores.keys()):
        scores = torch.stack(per_domain_scores[domain])
        domain_rows.append(
            {
                "domain_index": domain,
                "num_images": len(scores),
                "domain_accuracy": float(scores.mean().item()),
            }
        )

    domain_df = pd.DataFrame(domain_rows)

    if len(domain_df) == 0:
        raise RuntimeError("No domain scores were computed. Check the dataloader/test set.")

    #domain balanced score, the mean of the per domain means
    wada = float(domain_df["domain_accuracy"].mean())

    #image balanced score, just the mean over every image
    global_accuracy = float(torch.stack(all_image_scores).mean().item())

    domain_df.to_csv(output_dir / "wada_per_domain.csv", index=False)

    summary = {
        "wada_domain_mean": wada,
        "global_image_mean_accuracy": global_accuracy,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "num_domains": int(len(domain_df)),
        "num_images": int(sum(domain_df["num_images"])),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "wada_summary.csv", index=False)

    print("\nWADA/ADA-style test results")
    print("===========================")
    print(domain_df.to_string(index=False))
    print("")
    print(f"WADA/domain-balanced mean: {wada:.6f}")
    print(f"Global image mean accuracy: {global_accuracy:.6f}")
    print(f"Saved per-domain results to: {output_dir / 'wada_per_domain.csv'}")
    print(f"Saved summary to: {output_dir / 'wada_summary.csv'}")

    return domain_df, summary
  
@torch.no_grad()
def evaluate_map(
    detector,
    dataloader,
    device,
):
    """global map over one whole split

    map@50 is the number the thesis actually cares about, this also does a per
    domain map@50 breakdown, printed here and handed back as a second value for
    --eval_map to write to its own csv, it's worked out the same way the
    baseline and grl scripts do it in on_validation_epoch_end, a fresh single
    image map@50 per image, appended to that image's domain, then averaged
    within the domain, it's a slightly odd way to do it but two methods already
    use it, and matching it is the only way the coral per domain numbers stay
    comparable instead of being a third differently computed metric, assumes
    batch_size=1, which the val and test dataloaders here already use

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

        images = [
            image.to(device)
            for image in batch[0]
        ]

        targets = []

        for boxes in batch[1]:
            boxes = boxes.float().to(device)

            targets.append({
                "boxes": boxes,
                "labels": torch.ones(
                    len(boxes),
                    dtype=torch.long,
                    device=device,
                ),
            })

        predictions = detector(images)

        metric.update(
            predictions,
            targets,
        )

        #per domain breakdown, batch_size=1 so batch[2] is just the one
        #domain label for this image
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
        print(
            f"AP per class: "
            f"{results['map_per_class'].detach().cpu().tolist()}"
        )

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
        "map_mean_iou_10_50_75": float(
            results["map"].item()
        ),
        "map_50": float(
            results["map_50"].item()
        ),
        "map_75": float(
            results["map_75"].item()
        ),
    }, per_domain_rows
  
if __name__ == '__main__':

  args = parser_args()
  #take inputs from cli arguments

  SEED = args.seed
  print(f"SEED: {SEED}")

  #determinism block, copied verbatim out of mattia dutto's email (2026-08-06,
  #forwarded by petra bosilj 2026-08-12), same as in train_GWHD_baseline_clean.py
  #and train_GWHD_dgfrcnn_mattia.py, seeds random/numpy/lightning/torch and
  #flips the cudnn switches, all behind --deterministic, has to run before the
  #dataset, batch_sampler and model get built so SEED is right everywhere it
  #ends up, the sampler's seed= and coral's projection_seed=
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
  #take path to weights file

  weights_file = args.weights_file
  #take weight file name

  # Dataloader design based on input arguments
  # Training Dataset  
  
  
  #load datasets
  tr_dataset = WheatDataset('data/Annots/competition_train.csv', root_dir='data/gwhd_2021/images/', image_set = 'train', transform=train_transform) 
  vl_dataset = WheatDataset('data/Annots/competition_val.csv', root_dir='data/gwhd_2021/images/', image_set = 'val', transform=valid_transform)
  test_dataset = WheatDataset('data/Annots/competition_test.csv', root_dir='data/gwhd_2021/images/', image_set = 'tes', transform=valid_transform)
  
  #subsection of datasets for smokescreen training run
  tr_subdataset = Subset(tr_dataset, list(range(32)))
  vl_subdataset = Subset(vl_dataset, list(range(2)))


#we have two ways of instantiating
  if args.exp == 'coral':
    #cross domain coral is useless unless there's two domains in the batch
    #--sampler picks how that gets enforced, dg/samplers.py has the tradeoffs
    #between the three
    if args.sampler == 'domain_diverse':
        coral_batch_sampler = DomainDiverseBatchSampler(
            domain_labels=tr_dataset.domain_index.tolist(),
            batch_size=args.batch_size,
            seed=SEED,
        )
    elif args.sampler == 'domain_capped':
        coral_batch_sampler = DomainCappedBatchSampler(
            domain_labels=tr_dataset.domain_index.tolist(),
            batch_size=args.batch_size,
            domains_per_batch=args.domains_per_batch,
            seed=SEED,
        )
    else:
        coral_batch_sampler = NaturalDomainBatchSampler(
            domain_labels=tr_dataset.domain_index.tolist(),
            batch_size=args.batch_size,
            seed=SEED,
        )

    #in this case, we use that special dataloader during training
    train_dataloader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_sampler=coral_batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    #but if we want to run the training without using the coral modules, we use a different dataloader method, both come from the torch library though
  else:
    train_dataloader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
    )
#validation dataloader always has a batch size of 1
  val_dataloader = torch.utils.data.DataLoader(
      vl_dataset,
      batch_size=1,
      shuffle=False,
      collate_fn=collate_fn,
      num_workers=args.num_workers
  )
#same for test
  test_dataloader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=1,
      shuffle=False,
      collate_fn=collate_fn,
      num_workers=args.num_workers
  )

  # Instantiating the detector
  detector = DGFRCNN(
      2,
      args.batch_size,
      args.exp,
      args.reg_weights,
      lr=args.lr,
      weight_decay=args.weight_decay,
      optimizer_name=args.optimizer
  )
  
  if args.eval_map:

    ckpt_path = os.path.join(
        NET_FOLDER,
        weights_file + ".ckpt",
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )

    print(
        f"Loading checkpoint: {ckpt_path}"
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        #load checkpoint into cpu memory
    )

    detector.load_state_dict(
        checkpoint["state_dict"]
        #update detector with checkpoint weights
    )

    if args.eval_split == "val":
        eval_dataloader = val_dataloader
        split_name = "validation"
    else:
        eval_dataloader = test_dataloader
        split_name = "test"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nEvaluating {split_name} split..."
    )

    results, per_domain_rows = evaluate_map(
        detector=detector,
        dataloader=eval_dataloader,
        device=device,
    )

    output_path = os.path.join(
        NET_FOLDER,
        f"{weights_file}_{args.eval_split}_map.csv",
    )

    pd.DataFrame([{
        "split": args.eval_split,
        **results,
    }]).to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved results to {output_path}"
    )

    per_domain_output_path = os.path.join(
        NET_FOLDER,
        f"{weights_file}_{args.eval_split}_per_domain_map.csv",
    )

    pd.DataFrame([
        {"split": args.eval_split, **row} for row in per_domain_rows
    ]).to_csv(
        per_domain_output_path,
        index=False,
    )

    print(
        f"Saved per-domain results to {per_domain_output_path}"
    )

    sys.exit(0)
  
    #wada/ada style eval mode
  if args.eval_wada:
    ckpt_path = os.path.join(NET_FOLDER, weights_file + '.ckpt')

    if not os.path.exists(ckpt_path):
      raise FileNotFoundError(f"Could not find checkpoint for WADA evaluation: {ckpt_path}")

    print(f"Loading checkpoint for WADA evaluation: {ckpt_path}")
    detector.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])

    if args.eval_split == 'test':
      eval_dataloader = test_dataloader
      split_name = 'test'
    else:
      eval_dataloader = val_dataloader
      split_name = 'val'

    if args.wada_output_dir is not None:
      output_dir = args.wada_output_dir
    else:
      output_dir = os.path.join(NET_FOLDER, f"wada_{split_name}_{weights_file}")
    
    evaluate_wada(
      detector=detector,
      dataloader=eval_dataloader,
      output_dir=output_dir,
      score_threshold=args.score_threshold,
      iou_threshold=args.iou_threshold,
      max_batches=args.wada_max_batches
    )

    sys.exit(0)
  
  #train_dataloader = torch.utils.data.DataLoader(tr_subdataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True)
  #val_dataloader = torch.utils.data.DataLoader(vl_subdataset, batch_size=1, shuffle=False,  collate_fn=collate_fn, num_workers=0)
  
  #detector = DGFRCNN(2, 8, args.exp, args.reg_weights)

  if os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt'):
    detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  else:
    if not os.path.exists(NET_FOLDER):
      os.mkdir(NET_FOLDER, 0o777)

  #hand the batch_sampler and the output paths to the module so
  #on_train_epoch_end can write out the domain histogram and coral diag csvs
  if args.exp == 'coral':
    detector.coral_batch_sampler = coral_batch_sampler
    detector.domain_hist_csv_path = os.path.join(NET_FOLDER, f"{weights_file}_domain_histogram.csv")
    detector.coral_diag_csv_path = os.path.join(NET_FOLDER, f"{weights_file}_coral_diag.csv")

  early_stop_callback= EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')
  #save_last=True drops a NET_FOLDER/last.ckpt every epoch on top of the best
  #val_acc one under weights_file, same as the other two training scripts, that's
  #what lets an interrupted coral run pick back up rather than start over, which
  #matters when runpod reclaims a spot instance or the walltime cap hits
  checkpoint_callback = ModelCheckpoint(
      monitor='val_acc', dirpath=NET_FOLDER, filename=weights_file, mode='max',
      save_last=True,
  )
  #dirpath (NET_FOLDER) is shared by every seed of a repro run, run_repro_coral.sh and
  #run_coral_top2_seed_confirm.sh
  #passes the same --weights_folder each time and only changes --weights_file
  #with save_last=True that'd mean one generic last.ckpt sitting in a shared
  #folder, and the auto-resume check below would happily pick up the previous
  #seed's checkpoint instead of starting clean, so name the last checkpoint
  #after weights_file and keep each seed's resume state to itself
  checkpoint_callback.CHECKPOINT_NAME_LAST = f"{weights_file}-last"

  # trainer = Trainer(
  #    accelerator="cpu",
  #    devices=1,
  #    max_epochs=1,
  #    limit_train_batches=4,
  #    limit_val_batches=1,
  #    num_sanity_val_steps=0,
  #    callbacks=[checkpoint_callback],
  #    logger=False
  # )
  trainer = Trainer(
      accelerator="auto",
      max_epochs=args.max_epochs,
      deterministic=args.deterministic,
      accumulate_grad_batches=args.accumulate_grad_batches,
      callbacks=[checkpoint_callback, early_stop_callback],
      num_sanity_val_steps=2
  )

  #if this is a resubmission of a run that got cut off, walltime or a spot
  #instance getting reclaimed, pick up from the last checkpoint, restores the
  #whole trainer state, weights, optimizer, lr scheduler, epoch count and the
  #early stopping patience
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
  
  
  # detector.pr_file = 'pr_'+weights_file
  # detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  # trainer = Trainer(accelerator="gpu", max_epochs=0, num_sanity_val_steps=-1)
  # trainer.fit(detector, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
  #trainer.fit(detector, train_dataloaders=train_dataloader, val_dataloaders=train_dataloader)
  
  
