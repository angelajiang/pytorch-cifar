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
import os
import random

import gc

from models import *
from utils import progress_bar

import lib.backproppers
import lib.datasets
import lib.loggers
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
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-sched', default=None, help='Path to learning rate schedule')
    parser.add_argument('--momentum', default=0.9, type=float, help='learning rate')
    parser.add_argument('--decay', default=5e-4, type=float, help='decay')
    parser.add_argument('--resume-checkpoint-file', default=None, metavar='N',
                        help='checkpoint to resume from')
    parser.add_argument('--augment', '-a', dest='augment', action='store_true',
                        help='turn on data augmentation for CIFAR10')
    parser.add_argument('--kath', '-k', dest='kath', action='store_true',
                        help='Use Katharopoulous18 mode')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--net', default="resnet", metavar='N',
                        help='which network architecture to train')
    parser.add_argument('--dataset', default="cifar10", metavar='N',
                        help='which network architecture to train')
    parser.add_argument('--write-images', default=False, type=bool,
                        help='whether or not write png images by id')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for randomization; None to not set seed')
    parser.add_argument('--optimizer', default="sgd", metavar='N',
                        help='Optimizer among {sgd, adam}')
    parser.add_argument('--loss-fn', default="cross", metavar='N',
                        help='Loss function among {cross, hinge}')
    parser.add_argument('--loss-transformer-fn', default=None, metavar='N',
                        help='Transformation function among {loss_squared}')

    parser.add_argument('--sb-strategy', default="deterministic", metavar='N',
                        help='Selective backprop strategy among {baseline, deterministic, sampling}')
    parser.add_argument('--sb-start-epoch', type=int, default=0,
                        help='epoch to start selective backprop')
    parser.add_argument('--pickle-dir', default="/tmp/",
                        help='directory for pickles')
    parser.add_argument('--pickle-prefix', default="stats",
                        help='file prefix for pickles')
    parser.add_argument('--max-num-backprops', type=int, default=float('inf'), metavar='N',
                        help='how many images to backprop total')

    parser.add_argument('--sampling-strategy', default="square", metavar='N',
                        help='Selective backprop sampling strategy among {translate, nosquare, square}')
    parser.add_argument('--sampling-min', type=float, default=1,
                        help='Minimum sampling rate for sampling strategy')
    parser.add_argument('--sampling-max', type=float, default=1,
                        help='Maximum sampling rate for sampling strategy')
    parser.add_argument('--selectivity-scalar', type=float, default=1,
                        help='scale the select probability')

    # Logging and checkpointing interval
    parser.add_argument('--imageids-log-interval', type=int, default=10,
                        help='How often to write image ids to file (in epochs)')
    parser.add_argument('--losses-log-interval', type=int, default=10,
                        help='How often to write losses to file (in epochs)')
    parser.add_argument('--confidences-log-interval', type=int, default=10,
                        help='How often to write target confidences to file (in epochs)')
    parser.add_argument('--checkpoint-interval', type=int, default=None, metavar='N',
                        help='how often to save snapshot')

    parser.add_argument('--randomize-labels', type=float, default=None,
                        help='fraction of labels to randomize')
    parser.add_argument('--sample-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 1)')

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
                time.time()))

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


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_random_seeds(args.seed)

    # Model case
    print('==> Building model..')
    if args.net == "resnet":
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
                                       randomize_labels=args.randomize_labels)
    elif args.dataset == "mnist":
        dataset = lib.datasets.MNIST(device, args.test_batch_size)
    elif args.dataset == "svhn":
        dataset = lib.datasets.SVHN(net,
                                    args.test_batch_size,
                                    args.augment)
    else:
        print("Only cifar10, mnist, and svhn are implemented")
        exit()

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
    elif args.loss_fn == "hinge":
        loss_fn = nn.MultiMarginLoss
    else:
        print("Error: Loss function cannot be {}".format(args.loss_fn))
        exit()
    print("Using loss function {}".format(loss_fn))

    # Loss transformer function case
    if args.loss_transformer_fn == "loss_squared":
        loss_transformer_fn = lambda x: x * x
    else:
        loss_transformer_fn = lambda x: x
    print("Transforming loss function with {}".format(args.loss_transformer_fn))

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
    square = args.sampling_strategy in ["square", "translate"]
    translate = args.sampling_strategy in ["translate", "recenter"]

    probability_calculator = lib.selectors.SelectProbabiltyCalculator(args.sampling_min,
                                                                      args.sampling_max,
                                                                      len(dataset.classes),
                                                                      device,
                                                                      args.selectivity_scalar,
                                                                      square=square,
                                                                      translate=translate)
    if args.sb_strategy == "sampling":
        final_selector = lib.selectors.SamplingSelector(probability_calculator)
        final_backpropper = lib.backproppers.SamplingBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 loss_fn,
                                                                 loss_transformer_fn)
    elif args.sb_strategy == "deterministic":
        final_selector = lib.selectors.DeterministicSamplingSelector(probability_calculator,
                                                                     initial_sum=1)
        final_backpropper = lib.backproppers.SamplingBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 loss_fn,
                                                                 loss_transformer_fn)
    elif args.sb_strategy == "baseline":
        final_selector = lib.selectors.BaselineSelector()
        final_backpropper = lib.backproppers.BaselineBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 loss_fn,
                                                                 loss_transformer_fn)
    elif args.sb_strategy == "topk":
        final_selector = lib.selectors.TopKSelector(probability_calculator,
                                                    args.sample_size)
        final_backpropper = lib.backproppers.BaselineBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 loss_fn,
                                                                 loss_transformer_fn)
    elif args.sb_strategy == "lowk":
        final_selector = lib.selectors.LowKSelector(probability_calculator,
                                                    args.sample_size)
        final_backpropper = lib.backproppers.BaselineBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 loss_fn,
                                                                 loss_transformer_fn)
    elif args.sb_strategy == "randomk":
        final_selector = lib.selectors.RandomKSelector(probability_calculator,
                                                       args.sample_size)
        final_backpropper = lib.backproppers.BaselineBackpropper(device,
                                                                 dataset.model,
                                                                 optimizer,
                                                                 loss_fn,
                                                                 loss_transformer_fn)
    else:
        print("Use sb-strategy in {sampling, deterministic, baseline, topk, lowk, randomk}")
        exit()

    if args.kath:
        selector = None
        final_backpropper = lib.backproppers.ReweightedBackpropper(device,
                                                                   dataset.model,
                                                                   optimizer,
                                                                   loss_fn,
                                                                   loss_transformer_fn)
        backpropper = lib.backproppers.PrimedBackpropper(lib.backproppers.BaselineBackpropper(device,
                                                                                              dataset.model,
                                                                                              optimizer,
                                                                                              loss_fn,
                                                                                              loss_transformer_fn),
                                                         final_backpropper,
                                                         args.sb_start_epoch,
                                                         epoch=start_epoch)
        trainer = lib.trainer.KathTrainer(device,
                                          dataset.model,
                                          backpropper,
                                          args.batch_size,
                                          args.sample_size,
                                          max_num_backprops=args.max_num_backprops,
                                          lr_schedule=args.lr_sched)
    else:
        selector = lib.selectors.PrimedSelector(lib.selectors.BaselineSelector(),
                                                final_selector,
                                                args.sb_start_epoch,
                                                epoch=start_epoch)

        backpropper = lib.backproppers.PrimedBackpropper(lib.backproppers.BaselineBackpropper(device,
                                                                                              dataset.model,
                                                                                              optimizer,
                                                                                              loss_fn,
                                                                                              loss_transformer_fn),
                                                         final_backpropper,
                                                         args.sb_start_epoch,
                                                         epoch=start_epoch)
        trainer = lib.trainer.Trainer(device,
                                      dataset.model,
                                      selector,
                                      backpropper,
                                      args.batch_size,
                                      max_num_backprops=args.max_num_backprops,
                                      lr_schedule=args.lr_sched)

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
    trainer.on_forward_pass(logger.handle_forward_batch)
    trainer.on_backward_pass(logger.handle_backward_batch)
    trainer.on_backward_pass(image_id_hist_logger.handle_backward_batch)
    trainer.on_backward_pass(loss_hist_logger.handle_backward_batch)
    trainer.on_backward_pass(probability_by_image_logger.handle_backward_batch)
    stopped = False


    epoch = start_epoch

    while True:

        if stopped: break

        trainloader = torch.utils.data.DataLoader(dataset.trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        test(args, dataset, device, epoch, state, logger, loss_fn)

        trainer.train(trainloader)
        logger.next_partition()
        if trainer.stopped:
            stopped = True
            break

        logger.next_epoch()
        image_id_hist_logger.next_epoch()
        loss_hist_logger.next_epoch()
        probability_by_image_logger.next_epoch()
        if selector:
            selector.next_epoch()
        backpropper.next_epoch()
        epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser = set_experiment_default_args(parser)
    args = parser.parse_args()
    main(args)
