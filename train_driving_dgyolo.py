#!/usr/bin/python -tt

#Example run:
# python3 train_driving_dgfrcnn.py --exp dg --source_domains AC --target_domains A --weights_folder AC2A --weights_file ac2a_dgfrcnn --reg_weights 0.5 0.5 0.1 0.05 0.0001
#Here, A,B,C refer to the datasets ADCD, DCC100K and Cityscapes.   
#This command trains on datasets A and C and runs on dataset A.

from __future__ import absolute_import, division, print_function

import math, sys, time, random, os
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.autograd import Variable, Function

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

from DrivingDataset import *

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
seed_everything(SEED)


train_transform = A.Compose(
        [
        A.Resize(height=640, width=640, p=1.0),
        A.HorizontalFlip(p=0.5),     
        ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'])
    )

val_transform = A.Compose([
    A.Resize(height=640, width=640, p=1.0),
    ToTensorV2(p=1.0),
],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)
    

class _ImageDA(nn.Module):
    def __init__(self,dim,num_domains):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer 256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(256, 256, 3, stride=4)
        self.Conv2 = nn.Conv2d(256, 256, 3, stride=4)
        self.Conv3 = nn.Conv2d(256, 256, 3, stride=4)
        #self.Conv4 = nn.Conv2d(256, 256, 2, stride=1)
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        

        torch.nn.init.normal_(self.Conv1.weight, std=0.001)
        torch.nn.init.constant_(self.Conv1.bias, 0)
        torch.nn.init.normal_(self.Conv2.weight, std=0.001)
        torch.nn.init.constant_(self.Conv2.bias, 0)
        torch.nn.init.normal_(self.Conv3.weight, std=0.001)
        torch.nn.init.constant_(self.Conv3.bias, 0)
        #torch.nn.init.normal_(self.Conv4.weight, std=0.001)
        #torch.nn.init.constant_(self.Conv4.bias, 0)
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        #x=self.reLu(self.Conv4(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=F.sigmoid(self.linear2(x))
        
        return x

class _InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(_InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 64)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(64,self.num_domains)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsClsPrime(nn.Module):
    def __init__(self):
        super(_InsClsPrime,self).__init__()
        self.Conv1 = nn.Conv2d(256, 256, 1, stride=1)
        self.Conv2 = nn.Conv2d(256, 256, 1, stride=1)
        self.Conv3 = nn.Conv2d(256, 256, 1, stride=1)
        self.relu = nn.ReLU()
               

    def forward(self,x):
        y = []
        y.append(self.relu(self.Conv1(grad_reverse(x[0]))))
        y.append(self.relu(self.Conv2(grad_reverse(x[1]))))
        y.append(self.relu(self.Conv3(grad_reverse(x[2]))))

        return x

class _InsCls(nn.Module):
    def __init__(self, num_cls):
        super(_InsCls,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x
        
def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets=list()
    cls_labels = list()
    domain = list()
    
    for i, t, m, d in batch:
        images.append(i)
        targets.append(t)
        cls_labels.append(m)
        domain.append(d)
        
    images = torch.stack(images, dim=0)

    return images, targets, cls_labels, torch.tensor(domain)
    

from models.pytorchyolo.models import Darknet  
from models.pytorchyolo.utils.loss import compute_loss
from models.pytorchyolo.utils.utils import non_max_suppression

class DGYOLO(LightningModule):
    def __init__(self,n_classes, batch_size, exp, reg_weights):
        super(DGYOLO, self).__init__()
        self.n_classes = n_classes
        self.num_domains = len(tr_datasets)
        self.batch_size = batch_size
        self.exp = exp
        self.reg_weights = reg_weights
        
        self.detector = Darknet('./config/yolov3-custom-640.cfg')        
        self.detector.load_darknet_weights('./weights/darknet53.conv.74')
        self.ImageDA = _ImageDA(256, self.num_domains)
        self.InsDA = _InstanceDA(self.num_domains)
        
        
        self.base_lr = 0.1
        self.momentum = 0.9
        self.weight_decay=0.0005
        
        self.best_val_acc = 0
        self.log('val_acc', self.best_val_acc)
        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.5])        
                
        self.mode = 0
    
    def forward(self, imgs,targets=None):
      
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      
      optimizer = torch.optim.SGD([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay},
                                   {'params': self.ImageDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                   {'params': self.InsDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay}]) 
      
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08),
                      'monitor': 'val_acc'}
      
      
      return [optimizer], [lr_scheduler]
      
     
      
    def training_step(self, batch, batch_idx):
      
      #imgs = list(image.cuda() for image in batch[0]) 
      imgs = batch[0].cuda()
      
      
      targets = []
      for index, (boxes, labels) in enumerate(zip(batch[1], batch[2])):
        target = torch.zeros(boxes.shape[0], 6)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0] #w = xmax - xmin
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1] #h = ymax - ymin
        boxes[:, 0] = boxes[:, 0] + 0.5*boxes[:, 2] # xc = xmin + 0.5*w
        boxes[:, 1] = boxes[:, 1] + 0.5*boxes[:, 3] # yc = ymin + 0.5*h
        target[:, 2:] = boxes / 640.0 # Yolo bboxes are normalised between 0 and 1
        target[:, 1] = labels - 1 # Yolo labels start from zero: Verify this correctly
        target[:, 0] = index # Yolo uses the sample index in the first column
        targets.append(target)
      
      targets = torch.cat(targets, dim=0).cuda()

      if self.mode == 0:
        
        loss_dict = {}

        yolo_outputs, backbone_features, ins_feats = self.detector(imgs)
        loss, loss_components = compute_loss(yolo_outputs, targets, self.detector)
        loss_dict['detection_loss'] = loss
        
        if(self.exp == 'dg'):
          
          ImgDA_scores = self.ImageDA(backbone_features)
          loss_dict['DA_img_loss'] = self.reg_weights[0]*F.cross_entropy(ImgDA_scores, batch[3])
          
          ins_feat = ins_feats[-1]
          bs, c, y, x = ins_feat.shape
          ins_feat = ins_feat.view(bs, c, y*x).permute(0, 2, 1)
          IDA_out = self.InsDA(ins_feat)
          rep_factor = IDA_out.shape[1]
          ins_domain_labels = batch[3].reshape(self.batch_size,1).repeat(1, rep_factor)
          loss_dict['DA_ins_loss'] = self.reg_weights[1]*F.cross_entropy(IDA_out.permute(0, 2, 1), ins_domain_labels.to(device=0))
          
          
          ExpImgDA_scores = ImgDA_scores.clone().unsqueeze_(1).repeat(1, rep_factor, 1)
          
          loss_dict['Cst_loss'] = self.reg_weights[2]*F.mse_loss(IDA_out, ExpImgDA_scores)
          
          #self.mode = 1
              
      elif(self.mode == 1): #Without recording the gradients for detector, we need to update the weights for classifier weights
        
        loss_dict = {}
        loss = []
        #for index in range(self.num_domains):
          #for param in self.InsCls[index].parameters(): param.requires_grad = True
         
        for index in range(len(imgs)):
          with torch.no_grad():
            yolo_outputs, backbone_features, ins_feats = self.detector(imgs[index].unsqueeze(dim=0))
          yolo_outputs[0].requires_grad = True
          yolo_outputs[1].requires_grad = True
          yolo_outputs[2].requires_grad = True
          targets_indi = targets[targets[:, 0] == index]
          targets_indi[:, 0] = 0
          loss_indi, loss_components = compute_loss(yolo_outputs, targets_indi, self.detector)
          
          loss.append(loss_indi) 

        loss_dict['cls'] = self.reg_weights[4]*(torch.sum(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 2
        
      elif(self.mode == 2): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
      
        loss_dict = {}
        loss = []
    
        for index in range(len(imgs)):
          yolo_outputs, _, ins_feats = self.detector(imgs[index].unsqueeze(dim=0))
          for index_y in range(len(yolo_outputs)):
            yolo_outputs[index_y] = grad_reverse(yolo_outputs[index_y])
          targets_indi = targets[targets[:, 0] == index]
          targets_indi[:, 0] = 0
          loss_indi, loss_components = compute_loss(yolo_outputs, targets_indi, self.detector)
          loss.append(loss_indi)
  	  
        loss_dict['cls_prime'] = self.reg_weights[3]*(torch.sum(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 3
        
      else: #For Mode 3
      
        loss_dict = {}
        loss = []
        
        #for index in range(self.num_domains):
          #for param in self.InsCls[index].parameters(): param.requires_grad = False
        
        for index in range(len(imgs)):
          yolo_outputs, backbone_features, ins_feats = self.detector(imgs[index].unsqueeze(dim=0))
          
          for i in range(self.num_domains):
            if(i != batch[3][index].item()):
              #cls_scores = self.InsCls[i](self.box_features)
              targets_indi = targets[targets[:, 0] == index]
              targets_indi[:, 0] = 0
              loss_indi, loss_components = compute_loss(yolo_outputs, targets_indi, self.detector)
              loss.append(loss_indi)

        loss_dict['cls'] = self.reg_weights[4]*(torch.sum(torch.stack(loss))) 
        loss = sum(loss for loss in loss_dict.values())
        
        self.mode = 0
     		 
      return {"loss": loss}#, "log": torch.stack(temp_loss).detach().cpu()


    def validation_step(self, batch, batch_idx):
      
      img, boxes, labels, domain = batch
      
      targets = []
      for boxes, labels in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda() #/ 640
        target["labels"] = labels.long().cuda() - 1
        targets.append(target)
        
      dets = self.forward(img)
      dets = non_max_suppression(dets)
      preds = []
      for det in dets:
          pred = {}
          det = det[det[:, 4] > 0.2]
          boxes = det[:, :4]
          #boxes[:, 0] = boxes[:, 0] - 0.5*boxes[:, 2] # xmin = xc - 0.5*w
          #boxes[:, 1] = boxes[:, 1] - 0.5*boxes[:, 3] # ymin = yc - 0.5*h
          #boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # xmax = xmin + w
          #boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # ymax = ymin + h
          pred['boxes'] = torch.clip(boxes, min=0, max=640)
          pred['scores'] = det[:, 4].cuda()
          pred['labels'] = det[:, 5].to(torch.int32).cuda()
          preds.append(pred)
      
      
      try:
        self.metric.update(preds, targets)
      except:
        print(targets)
          
    def on_validation_epoch_end(self):
      
      metrics = self.metric.compute()
      
      self.log('val_acc', metrics['map_50'])
      print(metrics['map_per_class'], metrics['map_50'])
      self.metric.reset()


def parser_args():
  parser = argparse.ArgumentParser(description='DGFRCNN Main Experiments')
  parser.add_argument('--exp', dest='exp',
                      help='non_dg or dg',
                      default='non_dg', type=str)
                      
  parser.add_argument('--source_domains', dest='source_domains',
                      help='Source Domains provided as a string',
                      default='ABC', type=str)
                      
  parser.add_argument('--target_domains', dest='target_domains',
                      help='Target domains provided as string',
                      default='I', type=str)
  
  parser.add_argument('--weights_folder', dest='weights_folder',
                      help='Name of the weights folder',
                      default='ABC2I', type=str)
                      
  parser.add_argument('--weights_file', dest='weights_file',
                      help='Name of the weights file',
                      default='single_source_acdc', type=str)

  parser.add_argument('--reg_weights', nargs = 5, metavar=('a', 'b', 'c', 'd', 'e'), 
                       dest='reg_weights', help='Regularisation constats', type=float)
                      
  return parser.parse_args()
  
  
if __name__ == '__main__':

  args = parser_args()
  
  NET_FOLDER = args.weights_folder
  
  weights_file = args.weights_file  

  # Dataloader design based on input arguments
  # Training Dataset  
  tr_datasets = []
  domain_index = -1
  if 'a' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('data/Annots/acdc_train_all.csv', root='data/ACDC/rgb_anon/', transform=train_transform, domain=domain_index))
  if 'b' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('data/Annots/bdd10k_train_all.csv', root='data/BDD100K/images/10k/train/', transform=train_transform, domain=domain_index))
  if 'c' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('data/Annots/cityscapes_train_all.csv', root='data/Cityscapes/leftImg8bit/train/', transform=train_transform, domain=domain_index))
  if 'i' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('data/Annots/idd_train_all.csv', root='data/IDD/leftImg8bit/train/', transform=train_transform, domain=domain_index))
  
  tr_dataset = torch.utils.data.ConcatDataset(tr_datasets) # Combine all the source domains with their respective domain_index for training
    
  # Validation Dataset
  vl_datasets = []
  domain_index = -1
  if 'a' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('data/Annots/acdc_val_all.csv', root='data/ACDC/rgb_anon/', transform=val_transform, domain=domain_index))
  if 'b' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('data/Annots/bdd10k_val_all.csv', root='data/BDD100K/images/10k/val/', transform=val_transform, domain=domain_index))
  if 'c' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('data/Annots/cityscapes_val_all.csv', root='data/Cityscapes/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  if 'i' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('data/Annots/idd_val_all.csv', root='data/IDD/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  
  vl_dataset = torch.utils.data.ConcatDataset(vl_datasets) # Combine all the source domains with their respective domain_index for validation
  
  # Test Dataset
  test_datasets = []
  domain_index = -1
  if 'a' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('data/Annots/acdc_val_all.csv', root='data/ACDC/rgb_anon/', transform=val_transform, domain=domain_index))
  if 'b' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('data/Annots/bdd10k_val_all.csv', root='data/BDD100K/images/10k/val/', transform=val_transform, domain=domain_index))
  if 'c' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('data/Annots/cityscapes_val_all.csv', root='data/Cityscapes/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  if 'i' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('data/Annots/idd_val_all.csv', root='data/IDD/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  
  test_dataset = torch.utils.data.ConcatDataset(test_datasets) # Combine all the source domains with their respective domain_index for Testing


  train_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=16, drop_last=True)	
  val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  
  # Instantiating the detector
  detector = DGYOLO(9, 8, args.exp, args.reg_weights) # Num classes + 1 and batch_size

  if os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt'): 
    detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  else:	
    if not os.path.exists(NET_FOLDER):
      os.mkdir(NET_FOLDER, 0o777)

  
  early_stop_callback= EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')
  checkpoint_callback = ModelCheckpoint(monitor='val_acc', dirpath=NET_FOLDER, filename=weights_file, mode='max')
  
  trainer = Trainer(accelerator="gpu", max_epochs=100, deterministic=False, callbacks=[checkpoint_callback, early_stop_callback], num_sanity_val_steps=2)
  trainer.fit(detector, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
  
  detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  trainer = Trainer(accelerator="gpu", max_epochs=0, num_sanity_val_steps=-1)
  trainer.fit(detector, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
   
