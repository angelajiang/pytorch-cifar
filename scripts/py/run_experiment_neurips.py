import argparse
import json
import os
import subprocess


def set_experiment_default_args(parser):
    parser.add_argument('--expname', '-e', default="tmp", type=str, help='experiment name')
    parser.add_argument('--strategy', '-s', default="baseline", type=str, help='baseline, sb')
    parser.add_argument('--dataset', '-d', default="cifar10", type=str, help='mnist, cifar10, svhn, imagenet')
    parser.add_argument('--network', '-n', default="mobilenetv2", type=str, help='network architecture')
    parser.add_argument('--gradual', '-g', dest='gradual', action='store_true',
                        help='is learning rate gradual')
    parser.add_argument('--fast', '-f', dest='fast', action='store_true',
                        help='is learning rate accelerated')
    parser.add_argument('--forwardlr', dest='forwardlr', action='store_true',
                        help='learning rate schedule is based on forward props')

    parser.add_argument('--num-trials', default=1, type=int, help='number of trials')
    parser.add_argument('--src-dir', default="./", type=str, help='/path/to/pytorch-cifar')
    parser.add_argument('--dst-dir', default="/proj/BigLearning/ahjiang/output/", type=str, help='/path/to/dst/dir')
    return parser

def get_lr_sched_path(src_dir, dataset, gradual, fast):
    filename = "lrsched_{}".format(dataset)
    if gradual:
        filename += "_{}".format("gradual")
    else:
        filename += "_{}".format("step")
    if fast:
        filename += "_{}".format("fast")
    path = os.path.join(src_dir, "data/config/neurips", filename)
    return path

def get_max_num_backprops(lr_filename):
    with open(lr_filename) as f:
        data = json.load(f)
    last_lr_jump = max([int(k) for k in data.keys()])
    return int(last_lr_jump * 1.4)

def get_sampling_min(strategy):
    if strategy == "sb":
        return 0
    elif strategy == "baseline":
        return 1
    else:
        print("{} not a strategy".format(strategy))
        exit()


def get_batch_size():
    return 128

def get_decay():
    return 0.0005


class Seeder():
    def __init__(self):
        self.seed = 1336

    def get_seed(self):
        self.seed += 1
        return self.seed

def get_start_epoch():
    return 1337

def get_max_history_length():
    return 1024

def get_output_dirs(dst_dir):
    pickles_dir = os.path.join(dst_dir, "pickles")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if not os.path.exists(pickles_dir):
        os.mkdir(pickles_dir)
    return dst_dir, pickles_dir

def get_output_files(sb_strategy, dataset, net, sampling_min, batch_size, max_history_length, decay, trial, seed):
    output_file = "{}_{}_{}_{}_{}_{}_{}_trial{}_seed{}_v2".format(sb_strategy,
                                                               dataset,
                                                               net,
                                                               sampling_min,
                                                               batch_size,
                                                               max_history_length,
                                                               decay,
                                                               trial,
                                                               seed)

    pickle_file = "{}_{}_{}_{}_{}_{}_{}_trial{}_seed{}".format(sb_strategy,
                                                               dataset,
                                                               net,
                                                               sampling_min,
                                                               batch_size,
                                                               max_history_length,
                                                               decay,
                                                               trial,
                                                               seed)
    return output_file, pickle_file

def get_experiment_dirs(dst_dir, dataset, expname):
    output_dir = os.path.join(dst_dir, dataset, expname)
    pickles_dir = os.path.join(output_dir, "pickles")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(pickles_dir):
        os.mkdir(pickles_dir)
    return output_dir, pickles_dir

def get_sb_strategy():
    return "sampling"

def main(args):
    seeder = Seeder()
    src_dir = os.path.abspath(args.src_dir)
    lr_sched_path = get_lr_sched_path(src_dir, args.dataset, args.gradual, args.fast)
    if not os.path.isfile(lr_sched_path):
        print("{} is not a file").format(lr_sched_path)
        exit()
    max_num_backprops = get_max_num_backprops(lr_sched_path)
    sampling_min = get_sampling_min(args.strategy)
    batch_size = get_batch_size()
    decay = get_decay()
    start_epoch = get_start_epoch()
    output_dir, pickles_dir = get_experiment_dirs(args.dst_dir, args.dataset, args.expname)
    max_history_length = get_max_history_length()
    sb_strategy = get_sb_strategy()

    for trial in range(1, args.num_trials+1):
        seed = seeder.get_seed()
        output_file, pickle_file = get_output_files(sb_strategy,
                                                    args.dataset,
                                                    args.network,
                                                    sampling_min,
                                                    batch_size,
                                                    max_history_length,
                                                    decay,
                                                    trial,
                                                    seed)
        cmd = "python main.py "
        cmd += "--prob-strategy=relative-cubed "
        cmd += "--max-history-len={} ".format(max_history_length)
        cmd += "--dataset={} ".format(args.dataset)
        cmd += "--prob-loss-fn={} ".format("cross")
        cmd += "--sb-start-epoch={} ".format(start_epoch)
        cmd += "--sb-strategy={} ".format(sb_strategy)
        cmd += "--net={} ".format(args.network)
        cmd += "--batch-size={} ".format(batch_size)
        cmd += "--decay={} ".format(decay)
        cmd += "--max-num-backprops={} ".format(max_num_backprops)
        cmd += "--pickle-dir={} ".format(pickles_dir)
        cmd += "--pickle-prefix={} ".format(pickle_file)
        cmd += "--sampling-min={} ".format(sampling_min)
        cmd += "--seed={} ".format(seed)
        cmd += "--lr-sched={} ".format(lr_sched_path)
        cmd += "--augment"

        output_path = os.path.join(output_dir, output_file)
        print("========================================================================")
        print(cmd)
        print("------------------------------------------------------------------------")
        print(output_path)

        with open(os.path.join(pickles_dir, output_file) + "_cmd", "w+") as f:
            f.write(cmd)

        cmd_list = cmd.split(" ")
        with open(output_path, "w+") as f:
            subprocess.call(cmd_list, stdout=f)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser = set_experiment_default_args(parser)
    args = parser.parse_args()
    main(args)
