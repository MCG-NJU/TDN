# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import argparse
parser = argparse.ArgumentParser(
    description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', default=0.5, type=float, metavar='DO',
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int,
                    help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None,
                    help='fine-tune from checkpoint')
parser.add_argument('--experiment_name', type=str, default='TDN')
# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr_scheduler', type=str, default='step')
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--warmup_multiplier', type=int, default=100)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+", metavar='LRSteps',
                    help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float, metavar='W',
                    help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=True, action="store_true")
# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=5, type=int, metavar='N',
                    help='evaluation frequency (default: 5)')
# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--dense_sample', default=False, action="store_true",
                    help='use dense sample for video dataset')
parser.add_argument("--local_rank", type=int,
                    help='local rank for DistributedDataParallel')
