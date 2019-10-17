'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import cPickle as pickle
import pprint as pp
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models
import os
import random

import gc

from models import *
from utils import progress_bar

from lib.SelectiveBackpropper import SelectiveBackpropper
import lib.backproppers
import lib.calculators
import lib.datasets
import lib.forwardproppers
import lib.loggers
import lib.losses
import lib.selectors
import lib.trainer

BIAS_LOG_INTERVAL = 10
start_time_seconds = time.time()

def count_tensors():
    num_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                num_tensors += 1
        except:
            pass
    return num_tensors

def set_random_seeds(seed):
    if seed:
        print("Setting static random seeds to {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    return

def set_experiment_default_args(parser):
    strategy_options = ['nofilter', 'sb', 'kath', 'logbias', 'logaugment']
    calculator_options = ['relative', 'random', 'hybrid']
    fp_selector_options = ['alwayson', 'stale']

    # Basic training options
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-sched', default=None, help='Path to learning rate schedule')
    parser.add_argument('--momentum', default=0.9, type=float, help='learning rate')
    parser.add_argument('--decay', default=5e-4, type=float, help='decay')
    parser.add_argument('--resume-checkpoint-file', default=None, metavar='N',
                        help='checkpoint to resume from')
    parser.add_argument('--augment', '-a', dest='augment', action='store_true',
                        help='turn on data augmentation for CIFAR10')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--forward-batch-size', type=int, default=128, metavar='N',
                        help='batch size for informative forward pass')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--net', default="resnet", metavar='N',
                        help='which network architecture to train')
    parser.add_argument('--dataset', default="cifar10", metavar='N',
                        help='which network architecture to train')
    parser.add_argument('--datadir', default="./", metavar='N',
                        help='path to directory for ImageData loader')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for randomization; None to not set seed')
    parser.add_argument('--optimizer', default="sgd", metavar='N',
                        help='Optimizer among {sgd, adam}')
    parser.add_argument('--loss-fn', default="cross", metavar='N',
                        help='Loss function among {cross, hinge, cross_squared, cross_custom}')
    parser.add_argument('--max-num-backprops', type=int, default=float('inf'), metavar='N',
                        help='how many images to backprop total')

    # SB options
    parser.add_argument('--strategy', default='nofilter', choices=strategy_options)
    parser.add_argument('--calculator', default='relative', choices=calculator_options)
    parser.add_argument('--fp_selector', default='alwayson', choices=fp_selector_options)
    parser.add_argument('--sb-start-epoch', type=float, default=0,
                        help='epoch to start selective backprop')
    parser.add_argument('--prob-pow', type=float, default=1, metavar='N',
                        help='Power to scale probability by')
    parser.add_argument('--staleness', type=int, default=2,
                        help='Number of epochs to use stale losses for fp_selector')
    parser.add_argument('--kath-oversampling-rate', type=int, default=3, metavar='N',
                        help='how much to oversample by when running kath')
    parser.add_argument('--std-multiplier', type=float, default=1, metavar='N',
                        help='stdev multiplier for forward pass prob calculator')
    parser.add_argument('--sample-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--sampling-strategy', default="square", metavar='N',
                        help='Selective backprop sampling strategy among {nosquare, square}')
    parser.add_argument('--sampling-min', type=float, default=0,
                        help='Minimum sampling rate for sampling strategy')
    parser.add_argument('--sampling-max', type=float, default=1,
                        help='Maximum sampling rate for sampling strategy')
    parser.add_argument('--selectivity-scalar', type=float, default=1,
                        help='scale the select probability')
    parser.add_argument('--forwardlr', dest='forwardlr', action='store_true',
                        help='LR schedule based on forward passes')

    # Logging and checkpointing interval
    parser.add_argument('--no-logging', dest='no_logging', action='store_true',
                        help='turn off unnecessary logging')
    parser.add_argument('--pickle-dir', default="/tmp/",
                        help='directory for pickles')
    parser.add_argument('--pickle-prefix', default="stats",
                        help='file prefix for pickles')
    parser.add_argument('--imageids-log-interval', type=int, default=10,
                        help='How often to write image ids to file (in epochs)')
    parser.add_argument('--losses-log-interval', type=int, default=10,
                        help='How often to write losses to file (in epochs)')
    parser.add_argument('--confidences-log-interval', type=int, default=10,
                        help='How often to write target confidences to file (in epochs)')
    parser.add_argument('--checkpoint-interval', type=int, default=None, metavar='N',
                        help='how often to save snapshot')
    parser.add_argument('--log-bias', dest='log_bias', action='store_true',
                        help='Log bias by epoch')

    # Random features
    parser.add_argument('--randomize-labels', type=float, default=None,
                        help='fraction of labels to randomize')
    parser.add_argument('--write-images', default=False, type=bool,
                        help='whether or not write png images by id')

    return parser

def print_config(args):
    print("config sb-start-epoch {}".format(args.sb_start_epoch))
    print("config lr {}".format(args.lr))
    print("config lr-sched {}".format(args.lr_sched))
    print("config momentum {}".format(args.momentum))
    print("config decay {}".format(args.decay))
    print("config batch-size {}".format(args.batch_size))
    print("config net {}".format(args.net))
    print("config dataset {}".format(args.dataset))
    print("config seed {}".format(args.seed))
    print("config optimizer {}".format(args.optimizer))
    print("config loss-fn {}".format(args.loss_fn))
    print("config strategy {}".format(args.strategy))
    print("config calculator {}".format(args.calculator))
    print("config sampling-min {}".format(args.sampling_min))
    print("config sampling-max {}".format(args.sampling_max))
    print("config prob_pow {}".format(args.prob_pow))
    print("config forwardlr {}".format(args.forwardlr))

def test_sb(cnn, loader, epoch, sb):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0.
    for images, labels, ids in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)
            loss = nn.CrossEntropyLoss()(pred, labels)
            test_loss += loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    test_loss /= total
    val_acc = correct / total

    print('test_debug,{},{},{},{:.6f},{:.6f},{},{}'.format(
                epoch,
                sb.logger.global_num_backpropped,
                sb.logger.global_num_skipped,
                test_loss,
                100.*val_acc,
                sb.logger.global_num_skipped_fp,
                time.time() - start_time_seconds))
    cnn.train()
    return 100. * val_acc

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == "cuda"

    set_random_seeds(args.seed)

    # Model case
    print('==> Building model..')
    if args.net == "resnet":
        if args.dataset == "imagenet":
            net = torchvision.models.__dict__["resnet18"]()
        else:
            net = ResNet18()
    elif args.net == "vgg":
        net = VGG('VGG19')
    elif args.net == "preact_resnet":
        net = PreActResNet18()
    elif args.net == "googlenet":
        net = GoogLeNet()
    elif args.net == "densenet":
        net = DenseNet121()
    elif args.net == "resnext":
        net = ResNeXt29_2x64d()
    elif args.net == "mobilenet":
        net = MobileNet()
    elif args.net == "mobilenetv2":
        net = MobileNetV2()
    elif args.net == "dpn":
        net = DPN92()
    elif args.net == "shufflenet":
        net = ShuffleNetG2()
    elif args.net == "senet":
        net = SENet18()
    else:
        net = ResNet18()
    net = net.to(device)

    # Device case
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Dataset case
    if args.dataset == "cifar10":
        dataset = lib.datasets.CIFAR10(net,
                                       args.test_batch_size,
                                       args.augment,
                                       #args.batch_size * 4,
                                       None,
                                       randomize_labels=args.randomize_labels)
    elif args.dataset == "mnist":
        dataset = lib.datasets.MNIST(
                                    #10000,
                                    None,
                                    args.test_batch_size)
    elif args.dataset == "svhn":
        dataset = lib.datasets.SVHN(net,
                                    args.test_batch_size,
                                    #100000,
                                    None,
                                    args.augment)
    elif args.dataset == "imagenet":
        traindir = os.path.join(args.datadir, "train")
        valdir = os.path.join(args.datadir, "val")
        dataset = lib.datasets.ImageNet(net,
                                        args.test_batch_size,
                                        traindir,
                                        valdir,
                                        100000)
    else:
        print("Only cifar10, mnist, svhn and imagenet are implemented")
        exit()

    print(dataset.num_training_images)
    print_config(args)

    # Optimizer case
    if args.optimizer == "sgd":
        optimizer = optim.SGD(dataset.model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(dataset.model.parameters(),
                              lr=args.lr,
                              weight_decay=args.decay)

    # Loss function case
    if args.loss_fn == "cross":
        loss_fn = nn.CrossEntropyLoss
    elif args.loss_fn == "cross_squared":
        loss_fn = lib.losses.CrossEntropySquaredLoss
    elif args.loss_fn == "cross_custom":
        loss_fn = lib.losses.CrossEntropyLoss
    elif args.loss_fn == "cross_regulated":
        loss_fn = lib.losses.CrossEntropyRegulatedLoss
    elif args.loss_fn == "cross_regulated_boosted":
        loss_fn = lib.losses.CrossEntropyRegulatedBoostedLoss
    elif args.loss_fn == "hinge":
        loss_fn = nn.MultiMarginLoss
    else:
        print("Error: Loss function cannot be {}".format(args.loss_fn))
        exit()

    num_images_to_prime = int(args.sb_start_epoch * dataset.num_training_images)

    sb = SelectiveBackpropper(net,
                              optimizer,
                              args.prob_pow,
                              args.batch_size,
                              args.lr_sched,
                              len(dataset.classes),
                              dataset.num_training_images,
                              args.forwardlr,
                              args.strategy,
                              args.kath_oversampling_rate,
                              args.calculator,
                              args.fp_selector,
                              args.staleness)

    eval_every_n = args.batch_size * 10
    last_global_num_backpropped = 0
    epoch = 0

    while True:
        for dataset_split in dataset.get_dataset_splits(first_split_size=num_images_to_prime):

            if not args.no_logging:
                if sb.logger.global_num_backpropped - last_global_num_backpropped > eval_every_n: 
                    test_sb(net, dataset.testloader, epoch, sb)
                    last_global_num_backpropped = sb.logger.global_num_backpropped

            dataset_sampler = torch.utils.data.SubsetRandomSampler(dataset_split)
            trainloader = torch.utils.data.DataLoader(dataset.trainset,
                                                      batch_size=args.batch_size,
                                                      sampler=dataset_sampler,
                                                      num_workers=2)

            sb.trainer.train(trainloader)

            sb.next_partition()
        test_sb(net, dataset.testloader, epoch, sb)
        sb.next_epoch()
        epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser = set_experiment_default_args(parser)
    args = parser.parse_args()
    main(args)
