# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
import pickle

# options
parser = argparse.ArgumentParser(description="TDN testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--modalities', type=str, default='RGB')
parser.add_argument('--archs', type=str, default='resnet50')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)
parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--clip_index', type=int, default=0)
parser.add_argument('--output_dir',type=str,default="./result_file_0605_center16_ssv2",help='directory for pkl')
args = parser.parse_args()

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


weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)


data_iter_list = []
net_list = []
modality_list = args.modalities.split(',')
arch_list = args.archs.split('.')

total_num = None
for this_weights, this_test_segments, test_file, modality, this_arch in zip(weights_list, test_segments_list, test_file_list, modality_list, arch_list):
    num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,modality)
    net = TSN(num_class, this_test_segments, modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain
              )

    checkpoint = torch.load(this_weights)
    try:
        net.load_state_dict(checkpoint['state_dict'])
    except:
        checkpoint = checkpoint['state_dict']

        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias',
                        }
        for k, v in replace_dict.items():
            if k in base_dict:
                base_dict[v] = base_dict.pop(k)

        net.load_state_dict(base_dict)


    input_size = net.scale_size if args.full_res else net.input_size
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.dataset, root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
                       new_length=5 if modality == "RGB" else 5,
                       modality=modality,
                       image_tmpl=prefix,
                       clip_index=args.clip_index,
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=args.dense_sample, ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
    )

    if args.gpus is not None:
        devices = [args.gpus[i%len(args.gpus)] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    data_gen = enumerate(data_loader)

    if total_num is None:
        total_num = len(data_loader.dataset)
    else:
        assert total_num == len(data_loader.dataset)

    data_iter_list.append(data_gen)
    net_list.append(net)


output0 = []

def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():
        i, data, label = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        start_time = time.time()
        data_in = data.view(-1, length*5, data.size(2), data.size(3))
        data_in = data_in.view(batch_size , num_crop, this_test_segments, length*5, data_in.size(2), data_in.size(3))
        data_in0 = data_in[:,0,:,:,:,:]
        data_in0 = data_in0.view(batch_size , 1, this_test_segments, length*5, data.size(2), data.size(3))
        rst0 = net(data_in0)
        rst0 = rst0.reshape(batch_size, 1, -1)
                

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst0 = F.softmax(rst0, dim=1)

        inference_time = time.time() - start_time
        rst0 = rst0.data.cpu().numpy().copy()

        rst0 = rst0.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return i, rst0,0,0, label, inference_time


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

top01 = AverageMeter()
top05 = AverageMeter()
total_inference_time = 0.0

for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    with torch.no_grad():
        if i >= max_num:
            break
        this_rst0_list = []
        this_label = None
        for n_seg, (_, (data, label)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
            rst = eval_video((i, data, label), net, n_seg, modality)
            total_inference_time += rst[5]

            this_rst0_list.append(rst[1])
            this_label = label
        assert len(this_rst0_list) == len(coeff_list)
        for i_coeff in range(len(this_rst0_list)):
            this_rst0_list[i_coeff] *= coeff_list[i_coeff]

        ensembled_predict0 = sum(this_rst0_list) / len(this_rst0_list)

        for p, g in zip(ensembled_predict0, this_label.cpu().numpy()):
            output0.append([p[None, ...], g])
        cnt_time = time.time() - proc_start_time
        prec01, prec05 = accuracy(torch.from_numpy(ensembled_predict0), this_label, topk=(1, 5))
        top01.update(prec01.item(), this_label.numel())
        top05.update(prec05.item(), this_label.numel())
        if i % 20 == 0:
            print('video {} done, total {}/{}, average {:.5f} sec/video, '
                  'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                              float(total_inference_time) / (i+1) / args.batch_size, top01.avg, top05.avg))

video_pred0 = [np.argmax(x[0]) for x in output0]
video_pred0_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output0]

video_labels = [x[1] for x in output0]

output_dir = args.output_dir#'./result_file_0605_center16_ssv2'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Store results matrix into {}".format(output_dir))
output0_filepath = os.path.join(output_dir, str(args.clip_index)+'_'+'crop0'+'.pkl')
with open(output0_filepath, 'wb') as f:
    pickle.dump(output0, f, pickle.HIGHEST_PROTOCOL)
