from __future__ import absolute_import, division, print_function

import math, sys, time, random, os
from tqdm.notebook import tqdm
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


SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
seed_everything(SEED)

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
        self.domain_index = annotations['domain_index'] = annotations['domain'].map(unique_indices)
       
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

    return images, targets, domain_labels, orig_img


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
    def __init__(self,num_domains):
        super(_ImageDA,self).__init__()
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(2048, 2048, kernel_size=3, stride=4)
        self.Conv2 = nn.Conv2d(2048, 2048, kernel_size=3, stride=4)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.maxpool(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=self.reLu(self.linear2(x))
        x=torch.sigmoid(self.linear3(x))
        
        return x
   
class _InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(_InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()

        self.classifer=nn.Linear(128,self.num_domains)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsClsPrime(nn.Module):
    def __init__(self, num_cls):
        super(_InsClsPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 64)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(64,self.num_cls)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsCls(nn.Module):
    def __init__(self, num_cls):
        super(_InsCls,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 64)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(64,self.num_cls)
        

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

import fcos
class DGFCOS(LightningModule):
    def __init__(self, n_classes, batchsize, exp, reg_weights):
        super(DGFCOS, self).__init__()
        self.n_classes = n_classes
        self.batchsize = batchsize
        self.exp = exp
        self.reg_weights = reg_weights
        self.num_domains = 18
                
        self.detector = fcos.fcos_resnet50_fpn(min_size=1024, max_size=1024, num_classes=self.n_classes, pretrained_backbone=True)
	
        self.ImageDA = _ImageDA(self.num_domains)
        self.InsDA = _InstanceDA(self.num_domains)       
        self.InsCls = nn.ModuleList([_InsCls(self.n_classes) for i in range(self.num_domains)])
        self.InsClsPrime = nn.ModuleList([_InsClsPrime(self.n_classes) for i in range(self.num_domains)])
        
        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.5])
        self.base_lr = 1e-4 #Original base lr is 1e-4
        self.momentum = 0.9
        self.weight_decay=0.0001
        
        self.detector.backbone.body.register_forward_hook(self.store_backbone_out)        
        self.detector.head.register_forward_hook(self.store_head_input)  #For instance level features
        
        self.mode = 0
        
    def store_backbone_out(self, module, input1, output):
      self.base_feat = output['2']  #Output is a dict at three levels. We consider the top level feature as representative of image level feature
    
    def store_head_input(self, module, input1, output):
      
      temp0 = torch.reshape(input1[0][0], (input1[0][0].shape[0], input1[0][0].shape[1], input1[0][0].shape[2]*input1[0][0].shape[3]))
      temp1 = torch.reshape(input1[0][1], (input1[0][1].shape[0], input1[0][1].shape[1], input1[0][1].shape[2]*input1[0][1].shape[3]))
      temp2 = torch.reshape(input1[0][2], (input1[0][2].shape[0], input1[0][2].shape[1], input1[0][2].shape[2]*input1[0][2].shape[3]))
      temp3 = torch.reshape(input1[0][3], (input1[0][3].shape[0], input1[0][3].shape[1], input1[0][3].shape[2]*input1[0][3].shape[3]))
      temp4 = torch.reshape(input1[0][4], (input1[0][4].shape[0], input1[0][4].shape[1], input1[0][4].shape[2]*input1[0][4].shape[3]))
      
      self.ins_feat = torch.cat((temp0, temp1, temp2, temp3, temp4), -1).permute(0, 2, 1)
      
      
    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training 
      # and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      
      optimizer = torch.optim.Adam([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay},
                                    {'params': self.ImageDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsCls.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsClsPrime.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay}
                                      ],) 
      
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08),
                      'monitor': 'val_acc'}
      
      
      return [optimizer], [lr_scheduler]
            
    def training_step(self, batch, batch_idx):
      
      imgs = list(image.cuda() for image in batch[0]) 
      

      targets = []
      for boxes, domain in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = torch.ones(len(target["boxes"])).long().cuda()
        targets.append(target)
      
      if(self.mode == 0):
        loss_dict = {}
 
        temp_dict = self.detector(imgs, targets)       
        loss_dict['detection'] = temp_dict['classification'] + temp_dict['bbox_regression'] + temp_dict['bbox_ctrness']
        
        if(self.exp == 'dg'):        
                   
            ImgDA_scores  = self.ImageDA(self.base_feat)
            loss_dict['DA_img_loss'] = self.reg_weights[0]*F.cross_entropy(ImgDA_scores, torch.tensor(batch[2]).cuda().long())
                    
            IDA_out = self.InsDA(self.ins_feat)
            loss_dict['DA_ins_loss'] = self.reg_weights[1]*F.cross_entropy(IDA_out.permute(0,2,1), torch.tensor(batch[2]).repeat(IDA_out.shape[1],1).permute(1,0).cuda())
            
            loss_dict['Cst_loss'] = self.reg_weights[2]*F.mse_loss(IDA_out, ImgDA_scores.unsqueeze(dim=-1).repeat(1, 1, IDA_out.shape[1]).permute(0, 2, 1))
            self.mode = 1
            
        loss = sum(loss for loss in loss_dict.values())  
    
        
      elif(self.mode == 1): #Without recording the gradients for detector, we need to update the weights for classifier weights
        loss_dict = {}
        loss = []

        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = True

        for index in range(len(imgs)):
          with torch.no_grad():
            temp_dict = self.detector([imgs[index]], [targets[index]])
          cls_logits = self.InsCls[batch[2][index].item()](self.ins_feat)
          loss.append(F.cross_entropy(cls_logits, temp_dict['gt_classes'], reduction="mean")) 

        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 2
        
      elif(self.mode == 2): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
        loss_dict = {}
        loss = []
    
        for index in range(len(imgs)):
          temp_dict = self.detector([imgs[index]], [targets[index]])
          cls_logits = self.InsClsPrime[batch[2][index].item()](self.ins_feat)
          loss.append(F.cross_entropy(cls_logits, temp_dict['gt_classes'], reduction="mean")) 
  	  
        loss_dict['cls_prime'] = self.reg_weights[3]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 3
        
      else: #For Mode 3
      
        loss_dict = {}
        loss = []
        consis_loss = []
        
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = False
        
        for index in range(len(imgs)):
          temp_dict = self.detector([imgs[index]], [targets[index]])
          temp = []
          for i in range(len(self.InsCls)):
            if(i != batch[2][index].item()):
              cls_logits = self.InsCls[i](self.ins_feat)
              loss.append(F.cross_entropy(cls_logits, temp_dict['gt_classes'], reduction="mean")) 

        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())
        
        self.mode = 0
       
 	 
      return {"loss": loss}

    def validation_step(self, batch, batch_idx):
      
      img, boxes, labels, domain = batch
      
      preds = self.forward(img)

      for index in range(len(preds)):
           preds[index]['boxes'] = preds[index]['boxes'][preds[index]['scores'] > 0.2]
           preds[index]['labels'] = preds[index]['labels'][preds[index]['scores'] > 0.2]
           preds[index]['scores'] = preds[index]['scores'][preds[index]['scores'] > 0.2]
      
      targets = []
      for boxes, domain in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().to(device = self.device)
        target["labels"] = torch.ones(len(target["boxes"])).long().to(device = self.device)
        targets.append(target)
      
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
  parser = argparse.ArgumentParser(description='DGFCOS Main Experiments')
  parser.add_argument('--exp', dest='exp',
                      help='non_dg or dg',
                      default='non_dg', type=str)
                      
  parser.add_argument('--weights_folder', dest='weights_folder',
                      help='Name of the weights folder',
                      default='GWHD', type=str)
                      
  parser.add_argument('--weights_file', dest='weights_file',
                      help='Name of the weights file',
                      default='baseline', type=str)

  parser.add_argument('--reg_weights', nargs = 5, metavar=('a', 'b', 'c', 'd', 'e'), 
                       dest='reg_weights', help='Regularisation constats', type=float)
                      
  return parser.parse_args()

if __name__ == '__main__':

  args = parser_args()
  
  NET_FOLDER = args.weights_folder
  
  weights_file = args.weights_file  

  # Dataloader design based on input arguments
  # Training Dataset  
  tr_dataset = WheatDataset('data/Annots/competition_train.csv', root_dir='data/gwhd_2021/images/', image_set = 'train', transform=train_transform)
  vl_dataset = WheatDataset('data/Annots/competition_val.csv', root_dir='data/gwhd_2021/images/', image_set = 'val', transform=valid_transform)
  test_dataset = WheatDataset('data/Annots/competition_test.csv', root_dir='data/gwhd_2021/images/', image_set = 'tes', transform=valid_transform)


  train_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=16, drop_last=True)	
  val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  
  # Instantiating the detector
  detector = DGFCOS(2, 8, args.exp, args.reg_weights) # Num classes + 1 and batch_size

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

