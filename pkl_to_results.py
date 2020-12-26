# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import argparse
import time
import os
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
import torch
import pickle
from tqdm import tqdm
import numpy as np
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


parser = argparse.ArgumentParser(description="TDN testing on the full validation set")
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--num_clips', type=int, default=10)
parser.add_argument('--multi_segments',type=bool,default=False)
parser.add_argument('--output_dir',type=str,default='./result_file_ssv1_8f')
parser.add_argument('--output_dir1',type=str,default='./result_file_ssv1_16f')
args = parser.parse_args()

output_dir = args.output_dir
output_dir1 = args.output_dir1
output_filepath = os.path.join(output_dir, '0'+'_'+'crop'+str(args.test_crops-1)+'.pkl')

with open(output_filepath, 'rb') as f:
    output_file = pickle.load(f)
    num_videos = len(output_file)
    num_classes = output_file[0][0].shape[1]

num_clips = args.num_clips
num_crops = args.test_crops
ens_pred_numpy = np.zeros((num_videos, num_classes))
ens_label_numpy = np.zeros((num_videos,))
if num_crops == 1 :
    for clip_index in range(num_clips):
            output_filepath = os.path.join(output_dir, str(clip_index)+'_'+'crop'+str(0)+'.pkl')
            with open(output_filepath, 'rb') as f:
                output_file = pickle.load(f)
                for i in tqdm(range(len(output_file))):
                    pred_numpy = output_file[i][0]
                    if(not args.multi_segments):
                        pred_numpy = F.softmax(torch.from_numpy(pred_numpy), dim=1)
                        pred_numpy = pred_numpy.numpy()
                    ens_pred_numpy[i, :] = np.maximum(ens_pred_numpy[i, :] ,0)+ np.maximum(pred_numpy[0, :] ,0)
                    label = output_file[i][1]
                    ens_label_numpy[i] = ens_label_numpy[i] + label
else :
    for clip_index in range(num_clips):
        for crop_index in range(num_crops):
            output_filepath = os.path.join(output_dir, str(clip_index)+'_'+'crop'+str(crop_index)+'.pkl')
            with open(output_filepath, 'rb') as f:
                output_file = pickle.load(f)
                for i in tqdm(range(len(output_file))):
                    pred_numpy = output_file[i][0]
                    if(not args.multi_segments):
                        pred_numpy = F.softmax(torch.from_numpy(pred_numpy), dim=1)
                        pred_numpy = pred_numpy.numpy()
                    ens_pred_numpy[i, :] = np.maximum(ens_pred_numpy[i, :] ,0)+ np.maximum(pred_numpy[0, :] ,0)
                    label = output_file[i][1]
                    ens_label_numpy[i] = ens_label_numpy[i] + label
if(args.multi_segments):
    for clip_index in range(num_clips):
        for crop_index in range(num_crops):
            output_filepath = os.path.join(output_dir1, str(clip_index)+'_'+'crop'+str(crop_index)+'.pkl')
            with open(output_filepath, 'rb') as f:
                output_file = pickle.load(f)
                for i in tqdm(range(len(output_file))):
                    pred_numpy = output_file[i][0]
                    ens_pred_numpy[i, :] = np.maximum(ens_pred_numpy[i, :] ,0)+ np.maximum(pred_numpy[0, :] ,0)
                    label = output_file[i][1]

ens_pred_numpy = ens_pred_numpy / (num_clips*num_crops)
ens_label_numpy = ens_label_numpy / (int(num_clips)*int(num_crops))
prec1, prec5 = accuracy(torch.from_numpy(ens_pred_numpy), torch.from_numpy(ens_label_numpy).type(torch.LongTensor), topk=(1, 5))
print('-----Evaluation is finished------')
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(prec1, prec5))
