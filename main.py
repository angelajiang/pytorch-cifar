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

import lib.backproppers
import lib.calculators
import lib.datasets
import lib.forwardproppers
import lib.loggers
import lib.losses
import lib.selectors
import lib.trainer

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
    parser.add_argument('--sb-start-epoch', type=float, default=0,
                        help='epoch to start selective backprop')
    parser.add_argument('--sb-strategy', default="sampling", metavar='N',
                        help='Selective backprop strategy among {baseline, deterministic, sampling}')
    parser.add_argument('--prob-strategy', default="alwayson", metavar='N',
                        help='Probability calculator among {alwayson, pscale, proportional, relative}')
    parser.add_argument('--prob-pow', type=float, default=1, metavar='N',
                        help='Power to scale probability by')
    parser.add_argument('--prob-loss-fn', default="cross", metavar='N',
                        help='Loss function among {cross, mse}')
    parser.add_argument('--max-history-len', type=int, default=None, metavar='N',
                        help='History length for relative prob calculator')
    parser.add_argument('--fp-prob-strategy', default="alwayson", metavar='N',
                        help='Forward pass probability calculator among {alwayson}')
    parser.add_argument('--std-multiplier', type=float, default=1, metavar='N',
                        help='stdev multiplier for forward pass prob calculator')
    parser.add_argument('--sample-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--sampling-strategy', default="square", metavar='N',
                        help='Selective backprop sampling strategy among {nosquare, square}')
    parser.add_argument('--sampling-min', type=float, default=1,
                        help='Minimum sampling rate for sampling strategy')
    parser.add_argument('--sampling-max', type=float, default=1,
                        help='Maximum sampling rate for sampling strategy')
    parser.add_argument('--selectivity-scalar', type=float, default=1,
                        help='scale the select probability')
    parser.add_argument('--forwardlr', dest='forwardlr', action='store_true',
                        help='LR schedule based on forward passes')
    parser.add_argument('--kath', '-k', dest='kath', action='store_true',
                        help='Use Katharopoulous18 mode')
    parser.add_argument('--kath-strategy', default='reweighted', type=str,
                        help='Katharopoulous18 mode in {biased, reweighted, baseline}')

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

    # Random features
    parser.add_argument('--randomize-labels', type=float, default=None,
                        help='fraction of labels to randomize')
    parser.add_argument('--write-images', default=False, type=bool,
                        help='whether or not write png images by id')

    return parser


def get_stat(data):
    stat = {}
    stat["average"] = np.average(data)
    stat["p50"] = np.percentile(data, 50)
    stat["p75"] = np.percentile(data, 75)
    stat["p90"] = np.percentile(data, 90)
    stat["max"] = max(data)
    stat["min"] = min(data)
    return stat

class State:

    def __init__(self, num_images,
                       pickle_dir,
                       pickle_prefix,
                       num_backpropped=0,
                       num_skipped=0):
        self.num_images_backpropped = num_backpropped
        self.num_images_skipped = num_skipped
        self.num_images_seen = num_backpropped + num_skipped
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix

        self.init_target_confidences()

    def init_target_confidences(self):
        self.target_confidences = {}

        target_confidences_pickle_dir = os.path.join(self.pickle_dir, "target_confidences")
        self.target_confidences_pickle_file = os.path.join(target_confidences_pickle_dir,
                                                           "{}_target_confidences.pickle".format(self.pickle_prefix))

        # Make images hist pickle path
        if not os.path.exists(target_confidences_pickle_dir):
            os.mkdir(target_confidences_pickle_dir)

    def update_target_confidences(self, epoch, confidences, results, num_images_backpropped):
        if epoch not in self.target_confidences.keys():
            self.target_confidences[epoch] = {"confidences": [], "results": []}
        self.target_confidences[epoch]["confidences"] += confidences
        self.target_confidences[epoch]["results"] += results
        self.target_confidences[epoch]["num_backpropped"] = num_images_backpropped

    def write_summaries(self):
        with open(self.target_confidences_pickle_file, "wb") as handle:
            print(self.target_confidences_pickle_file)
            pickle.dump(self.target_confidences, handle, protocol=pickle.HIGHEST_PROTOCOL)

def mini_test(args,
              dataset,
              device,
              epoch,
              state,
              logger,
              loss_fn):

    net = dataset.model
    testloader = dataset.testloader

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, image_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = loss_fn()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader.dataset)
    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                logger.global_num_backpropped,
                logger.global_num_skipped,
                test_loss,
                100.*correct/total,
                logger.global_num_skipped_fp))

def test(args,
         dataset,
         device,
         epoch,
         state,
         logger,
         loss_fn):

    net = dataset.model
    testloader = dataset.testloader

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if epoch % args.confidences_log_interval == 0:
        write_target_confidences = True
    else:
        write_target_confidences = False

    with torch.no_grad():
        for batch_idx, (inputs, targets, image_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = loss_fn()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if write_target_confidences:
                softmax_outputs = nn.Softmax()(outputs)
                targets_array = targets.cpu().numpy()
                outputs_array = softmax_outputs.cpu().numpy()
                confidences = [o[t] for t, o in zip(targets_array, outputs_array)]
                results = predicted.eq(targets).data.cpu().numpy().tolist()
                state.update_target_confidences(epoch,
                                                confidences,
                                                results,
                                                logger.global_num_backpropped)
    if write_target_confidences:
        state.write_summaries()

    test_loss /= len(testloader.dataset)
    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                logger.global_num_backpropped,
                logger.global_num_skipped,
                test_loss,
                100.*correct/total,
                logger.global_num_skipped_fp))

    # Save checkpoint.
    if args.checkpoint_interval:
        if epoch % args.checkpoint_interval == 0:
            acc = 100.*correct/total
            print('Saving..')
            net_state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'num_backpropped': logger.global_num_backpropped,
                'num_skipped': logger.global_num_skipped,
                'dataset': dataset,
            }
            checkpoint_dir = os.path.join(args.pickle_dir, "checkpoint")
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            checkpoint_file = os.path.join(checkpoint_dir,
                                           args.pickle_prefix + "_epoch{}_ckpt.t7".format(epoch))
            print("Saving checkpoint at {}".format(checkpoint_file))
            torch.save(net_state, checkpoint_file)

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
    print("config sb-strategy {}".format(args.sb_strategy))
    print("config prob-strategy {}".format(args.prob_strategy))
    print("config fp-prob-strategy {}".format(args.fp_prob_strategy))
    print("config prob-loss-fn {}".format(args.prob_loss_fn))
    print("config max-num-backprops {}".format(args.max_num_backprops))
    print("config sampling-min {}".format(args.sampling_min))
    print("config sampling-max {}".format(args.sampling_max))
    print("config prob_pow {}".format(args.prob_pow))
    print("config max-history_len {}".format(args.max_history_len))
    print("config forwardlr {}".format(args.forwardlr))
    print("config kath {}".format(args.kath))
    print("config kath-strategy {}".format(args.kath_strategy))

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Checkpointing case
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    start_num_backpropped = 0
    start_num_skipped = 0

    if args.resume_checkpoint_file:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print("Loading checkpoint at {}".format(args.resume_checkpoint_file))
        checkpoint = torch.load(args.resume_checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        start_num_backpropped = checkpoint['num_backpropped']
        start_num_skipped = checkpoint['num_skipped']
        if "dataset" in checkpoint.keys():
            dataset = checkpoint['dataset']

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

    if args.prob_loss_fn == "cross":
        prob_loss_fn = nn.CrossEntropyLoss
    elif args.prob_loss_fn == "mse":
        prob_loss_fn = lib.losses.MSELoss
    else:
        print("Error: Loss function cannot be {}".format(args.loss_fn))
        exit()


    # Miscellaneous setup

    state = State(dataset.num_training_images,
                  args.pickle_dir,
                  args.pickle_prefix,
                  start_num_backpropped,
                  start_num_skipped)

    if args.write_images:
        image_writer = lib.loggers.ImageWriter('./data', args.dataset, dataset.unnormalizer)
        for partition in dataset.partitions:
            image_writer.write_partition(partition)

    ## Setup Trainer ##

    # Setup Trainer: Calculator and Selector
    bp_probability_calculator = lib.calculators.get_probability_calculator(args.prob_strategy,
                                                                           device,
                                                                           prob_loss_fn,
                                                                           args.sampling_min,
                                                                           args.sampling_max,
                                                                           len(dataset.classes),
                                                                           args.max_history_len,
                                                                           args.prob_pow)

    fp_probability_calculator = lib.calculators.HistoricalProbabilityCalculator(args.fp_prob_strategy,
                                                                                args.std_multiplier,
                                                                                bp_probability_calculator)

    num_images_to_prime = int(args.sb_start_epoch * dataset.num_training_images)
    bp_selector = lib.selectors.get_selector(args.sb_strategy,
                                             bp_probability_calculator,
                                             num_images_to_prime,
                                             args.sample_size,
                                             forwards=False)
    fp_selector = lib.selectors.get_selector(args.sb_strategy,
                                             fp_probability_calculator,
                                             num_images_to_prime,
                                             args.sample_size,
                                             forwards=True)

    # Setup Trainer: Backpropper and Trainer
    if args.kath:
        selector = None
        if args.kath_strategy == "reweighted":
            final_backpropper = lib.backproppers.ReweightedBackpropper(device,
                                                                       dataset.model,
                                                                       optimizer,
                                                                       loss_fn)
        else:
            final_backpropper = lib.backproppers.SamplingBackpropper(device,
                                                                     dataset.model,
                                                                     optimizer,
                                                                     loss_fn)
        backpropper = lib.backproppers.PrimedBackpropper(lib.backproppers.SampilngBackpropper(device,
                                                                                              dataset.model,
                                                                                              optimizer,
                                                                                              loss_fn),
                                                         final_backpropper,
                                                         num_images_to_prime)
        if args.kath_strategy == "baseline":
            trainer = lib.trainer.KathBaselineTrainer(device,
                                                      dataset.model,
                                                      dataset,
                                                      backpropper,
                                                      args.batch_size,
                                                      args.sample_size,
                                                      loss_fn,
                                                      max_num_backprops=args.max_num_backprops,
                                                      lr_schedule=args.lr_sched)
        else:
            trainer = lib.trainer.KathTrainer(device,
                                              dataset.model,
                                              dataset,
                                              backpropper,
                                              args.batch_size,
                                              args.sample_size,
                                              loss_fn,
                                              max_num_backprops=args.max_num_backprops,
                                              lr_schedule=args.lr_sched,
                                              forwardlr=args.forwardlr)
    else:
        backpropper = lib.backproppers.SamplingBackpropper(device,
                                                           dataset.model,
                                                           optimizer,
                                                           loss_fn)
        forwardpropper = lib.forwardproppers.BaselineForwardpropper(device,
                                                                    dataset.model,
                                                                    dataset,
                                                                    optimizer,
                                                                    loss_fn)
        '''
        trainer = lib.trainer.Trainer(device,
                                      dataset.model,
                                      dataset,
                                      bp_selector,
                                      backpropper,
                                      args.batch_size,
                                      loss_fn,
                                      max_num_backprops=args.max_num_backprops,
                                      lr_schedule=args.lr_sched,
                                      forwardlr=args.forwardlr)
        '''
        trainer = lib.trainer.MemoizedTrainer(device,
                                              dataset.model,
                                              dataset,
                                              bp_selector,
                                              backpropper,
                                              args.batch_size,
                                              fp_selector,
                                              forwardpropper,
                                              args.forward_batch_size,
                                              loss_fn,
                                              max_num_backprops=args.max_num_backprops,
                                              lr_schedule=args.lr_sched,
                                              forwardlr=args.forwardlr)

    logger = lib.loggers.Logger(log_interval = args.log_interval,
                                epoch=start_epoch,
                                num_backpropped=start_num_backpropped,
                                num_skipped=start_num_skipped)
    image_id_hist_logger = lib.loggers.ImageIdHistLogger(args.pickle_dir,
                                                         args.pickle_prefix,
                                                         dataset.num_training_images,
                                                         args.imageids_log_interval)
    loss_hist_logger = lib.loggers.LossesByEpochLogger(args.pickle_dir,
                                                       args.pickle_prefix,
                                                       args.losses_log_interval)
    probability_by_image_logger = lib.loggers.ProbabilityByImageLogger(args.pickle_dir,
                                                                       args.pickle_prefix)

    if not args.no_logging:
        trainer.on_forward_pass(logger.handle_forward_batch)
        trainer.on_backward_pass(logger.handle_backward_batch)
        trainer.on_forward_mark(logger.handle_forward_mark)
        trainer.on_backward_pass(image_id_hist_logger.handle_backward_batch)
        trainer.on_backward_pass(loss_hist_logger.handle_backward_batch)
        trainer.on_backward_pass(probability_by_image_logger.handle_backward_batch)

    stopped = False
    epoch = start_epoch

    eval_every_n = args.batch_size * 10
    last_global_num_backpropped = 0

    while True:

        if stopped: break

        for dataset_split in dataset.get_dataset_splits(first_split_size=num_images_to_prime):

            if not args.no_logging:
                if logger.global_num_backpropped - last_global_num_backpropped > eval_every_n: 
                    mini_test(args, dataset, device, epoch, state, logger, loss_fn)
                    last_global_num_backpropped = logger.global_num_backpropped

            dataset_sampler = torch.utils.data.SubsetRandomSampler(dataset_split)
            trainloader = torch.utils.data.DataLoader(dataset.trainset,
                                                      batch_size=args.batch_size,
                                                      sampler=dataset_sampler,
                                                      num_workers=0)

            trainer.train(trainloader)
            logger.next_partition()
            if bp_selector:
                bp_selector.next_partition(len(dataset_split))
                fp_selector.next_partition(len(dataset_split))
            if trainer.stopped:
                stopped = True
                break

        test(args, dataset, device, epoch, state, logger, loss_fn)
        logger.next_epoch()

        if not args.no_logging:
            image_id_hist_logger.next_epoch()
            loss_hist_logger.next_epoch()
            probability_by_image_logger.next_epoch()

        epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser = set_experiment_default_args(parser)
    args = parser.parse_args()
    main(args)
