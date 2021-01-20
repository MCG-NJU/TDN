# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import os
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.logger import setup_logger
from ops.lr_scheduler import get_scheduler
from ops.utils import reduce_tensor
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import *
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    full_arch_name = args.arch
    args.store_name = '_'.join(['TDN_', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    if dist.get_rank() == 0:
        check_rootfolders()

    logger = setup_logger(output=os.path.join(args.root_log, args.store_name),
                          distributed_rank=dist.get_rank(),
                          name=f'TDN')
    logger.info('storing name: ' + args.store_name)

    model = TSN(num_class,
                args.num_segments,
                args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=(args.tune_from and args.dataset in args.tune_from))

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    for group in policies:
        logger.info(
            ('[TDN-{}]group: {} has {} params, lr_mult: {}, decay_mult: {}'.
             format(args.arch, group['name'], len(group['params']),
                    group['lr_mult'], group['decay_mult'])))

    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    train_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([train_augmentation,
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,]),
        dense_sample=args.dense_sample)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler,drop_last=True)  

    val_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,]),
        dense_sample=args.dense_sample)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True, sampler=val_sampler, drop_last=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.SGD(policies, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = DistributedDataParallel(model.cuda(), device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        logger.info(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                logger.info('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                logger.info('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        logger.info('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            logger.info('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)


    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    if args.evaluate:
        logger.info(("===========evaluate==========="))
        val_loader.sampler.set_epoch(args.start_epoch)
        prec1, prec5, val_loss = validate(val_loader, model, criterion, logger)
        if dist.get_rank() == 0:
            is_best = prec1 > best_prec1
            best_prec1 = prec1
            logger.info(("Best Prec@1: '{}'".format(best_prec1)))
            save_epoch = args.start_epoch + 1
            save_checkpoint(
                {
                    'epoch': args.start_epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'prec1': prec1,
                    'best_prec1': best_prec1,
                }, save_epoch, is_best)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch=epoch, logger=logger, scheduler=scheduler)
        if dist.get_rank() == 0:
            tf_writer.add_scalar('loss/train', train_loss, epoch)
            tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
            tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loader.sampler.set_epoch(epoch)
            prec1, prec5, val_loss = validate(val_loader, model, criterion, logger)
            if dist.get_rank() == 0:
                tf_writer.add_scalar('loss/test', val_loss, epoch)
                tf_writer.add_scalar('acc/test_top1', prec1, epoch)
                tf_writer.add_scalar('acc/test_top5', prec5, epoch)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

                logger.info(("Best Prec@1: '{}'".format(best_prec1)))
                tf_writer.flush()
                save_epoch = epoch + 1
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'prec1': prec1,
                        'best_prec1': best_prec1,
                    }, save_epoch, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger=None, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                             top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))  # TODO
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            output = model(input)

            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            loss = reduce_tensor(loss)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
        .format(top1=top1, top5=top5, loss=losses)))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, epoch, is_best):
    filename = '%s/%s/%d_epoch_ckpt.pth.tar' % (args.root_model, args.store_name, epoch)
    torch.save(state, filename)
    if is_best:
        best_filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, best_filename)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


if __name__ == '__main__':
    main()