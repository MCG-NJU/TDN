# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import numpy as np
import torch.distributed as dist
from sklearn.metrics import confusion_matrix


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print((name, grad_out[0].data.abs().mean(),
               grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))
        print((grad_out[0]))
        print((grad_in[0]))

    return hook


def log_add(log_a, log_b):
    return log_a + np.log(1 + np.exp(log_b - log_a))


def class_accuracy(prediction, label):
    cf = confusion_matrix(prediction, label)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt.astype(float)
    mean_cls_acc = cls_acc.mean()
    return cls_acc, mean_cls_acc


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
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
