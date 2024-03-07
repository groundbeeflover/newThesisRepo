#This file is the helper code used to convert all the annotations into csv format
import json
import cv2
import numpy as np
from os import walk
import pandas as pd
import random
import os, fnmatch
import argparse
from pathlib2 import Path

  
def encode_boxes(boxes):

  if len(boxes) >0:
    boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
    BoxesString = ";".join(boxes)
  else:
    BoxesString = "no_box"
  return BoxesString

def encode_labels(labels):

  if len(labels) >0:
    labels = [" ".join([str(item)]) for item in labels]
    LabelsString = ";".join(labels)
  else:
    LabelsString = "no_label"
  return LabelsString
  
train_df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])  
val_df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])

train_obj_freq = {1:0, 2:0, 3:0, 4:0, 5:0}
val_obj_freq = {1:0, 2:0, 3:0, 4:0, 5:0}

def parser_args():
  parser = argparse.ArgumentParser(description='Convert JSON2CSV')
  parser.add_argument('--category', dest='category',
                      help='real or synthetic',
                      default='all', type=str)
                      
  parser.add_argument('--synthetic_index', dest='index',
                      help='Valid values are between 1 to 7',
                      default='1', type=int)
                      
  args = parser.parse_args()
  return args

if __name__ == '__main__':     

  args = parser_args()
  
  if args.category == 'real':
      train_root_path = 'URPC2019/train2017'
      val_root_path = 'URPC2019/val2017'
      train_annots_path = 'URPC2019/train2017'
      val_annots_path = 'URPC2019/val2017'
  else:
      train_root_path = 'URPC2019/type'+str(args.index)
      val_root_path = 'URPC2019/val_type'+str(args.index)
      train_annots_path = 'URPC2019/train2017'
      val_annots_path = 'URPC2019/val2017'

  for image_path in Path(train_root_path).rglob('*.jpg'): 
      
      image_name  = str(image_path).split('/')[-1].split('.')[0]
      
      txt_path = train_annots_path + '/' + image_name + '.txt'
      
      img = cv2.imread(str(image_path))
      H, W, C = img.shape
      
      try:
        f = open(txt_path, 'r')
      except:
        BoxesString = encode_boxes([])
        LablesString = encode_labels([])
        
        new_row = {'image_name':str(image_path), 'BoxesString': BoxesString, 'LabelsString': LabelsString}
        
        train_df = pd.concat([train_df, pd.DataFrame.from_records([new_row])], ignore_index = True)
        
        continue
      
      
      bboxes = []
      labels = []
      for line in f.readlines():
          label, xc, yc, w, h = line.strip('\n').split(' ')
          
          xmin = (float(xc) - 0.5*float(w))*W
          ymin = (float(yc) - 0.5*float(h))*H
          xmax = (float(xc) + 0.5*float(w))*W
          ymax = (float(yc) + 0.5*float(h))*H
          
          bboxes.append([xmin, ymin, xmax, ymax])
          labels.append(int(label))
          
          train_obj_freq[int(label)] += 1 
       
      BoxesString = encode_boxes(bboxes)
      LabelsString = encode_labels(labels)
      
      new_row = {'image_name':str(image_path), 'BoxesString': BoxesString, 'LabelsString': LabelsString}
      
      train_df = pd.concat([train_df, pd.DataFrame.from_records([new_row])], ignore_index = True)
  
  for image_path in Path(val_root_path).rglob('*.jpg'): 
      
      image_name  = str(image_path).split('/')[-1].split('.')[0]
      
      txt_path = val_annots_path + '/' + image_name + '.txt'
      
      img = cv2.imread(str(image_path))
      H, W, C = img.shape
      
      try:
        f = open(txt_path, 'r')
      except:
        BoxesString = encode_boxes([])
        LablesString = encode_labels([])
        
        new_row = {'image_name':str(image_path), 'BoxesString': BoxesString, 'LabelsString': LabelsString}
        
        val_df = pd.concat([val_df, pd.DataFrame.from_records([new_row])], ignore_index = True)
        
        continue
      
      
      bboxes = []
      labels = []
      for line in f.readlines():
          label, xc, yc, w, h = line.strip('\n').split(' ')
          
          xmin = (float(xc) - 0.5*float(w))*W
          ymin = (float(yc) - 0.5*float(h))*H
          xmax = (float(xc) + 0.5*float(w))*W
          ymax = (float(yc) + 0.5*float(h))*H
          
          bboxes.append([xmin, ymin, xmax, ymax])
          labels.append(int(label))
          
          val_obj_freq[int(label)] += 1 
       
      BoxesString = encode_boxes(bboxes)
      LabelsString = encode_labels(labels)
      
      new_row = {'image_name':str(image_path), 'BoxesString': BoxesString, 'LabelsString': LabelsString}
      val_df = pd.concat([val_df, pd.DataFrame.from_records([new_row])], ignore_index = True)
      
      
  train_df = train_df.reset_index(drop=True)
  val_df = val_df.reset_index(drop=True)

  if args.category == 'real':
    train_df.to_csv('./Annots/urpc_train_real.csv')   
    val_df.to_csv('./Annots/urpc_val_real.csv')
  else:
    train_df.to_csv('./Annots/urpc_trainsyn_domain_'+str(args.index)+'.csv')
    val_df.to_csv('./Annots/urpc_valsyn_domain_'+str(args.index)+'.csv')


print(train_df.head())
print(val_df.head())
print(train_obj_freq)
print(val_obj_freq)
