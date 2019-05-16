import argparse
import json
import os
import subprocess


def set_experiment_default_args(parser):
    parser.add_argument('--expname', '-e', default="tmp", type=str, help='experiment name')
    parser.add_argument('--src-dir', default="./", type=str, help='/path/to/pytorch-cifar')
    parser.add_argument('--dst-dir', default="/proj/BigLearning/ahjiang/output/benchmarks", type=str, help='/path/to/dst/dir')
    return parser

def get_start_epoch():
    return 1000

def get_max_num_backprops():
    return 100000

def get_output_files(network, batch_size):
    output_file = "asymmetry_{}_{}".format(network, batch_size)
    return output_file

def get_experiment_dirs(dst_dir, expname):
    output_dir = os.path.join(dst_dir, expname)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir

def main(args):
    src_dir = os.path.abspath(args.src_dir)
    max_num_backprops = get_max_num_backprops()
    start_epoch = get_start_epoch()
    output_dir = get_experiment_dirs(args.dst_dir, args.expname)

    networks = ["resnet", "mobilenetv2"]
    batch_sizes = [32, 64, 128]

    for network in networks:
        for batch_size in batch_sizes:
            output_file  = get_output_files(network, batch_size)
            cmd = "python main.py "
            cmd += "--net={} ".format(network)
            cmd += "--batch-size={} ".format(batch_size)
            cmd += "--sb-start-epoch={} ".format(start_epoch)
            cmd += "--max-num-backprops={}".format(max_num_backprops)

            output_path = os.path.join(output_dir, output_file)
            print("========================================================================")
            print(cmd)
            print("------------------------------------------------------------------------")
            print(output_path)

            cmd_list = cmd.split(" ")
            with open(output_path, "w+") as f:
                subprocess.call(cmd_list, stdout=f)


if __name__ == '__main__':

    # git revert d9c2c017bd8eaaa2e97d5699c289b7dd04c48293
    # python scripts/py/benchmark_asymmetry_neurips.py -e 190516_asymmetry_titanv

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser = set_experiment_default_args(parser)
    args = parser.parse_args()
    main(args)
