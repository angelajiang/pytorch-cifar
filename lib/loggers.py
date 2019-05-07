import numpy as np
import pickle
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class BiasByEpochLogger(object):

    def __init__(self, pickle_dir, pickle_prefix, log_frequency):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.log_frequency = log_frequency
        self.pickle_prefix = pickle_prefix
        self.init_data()

    def next_epoch(self):
        self.write()
        self.current_epoch += 1
        self.data = self.base_dict(self.current_epoch)

    def base_dict(self, epoch):
        return {"epoch": epoch,
                "selectivities": [],
                "cos_sims": [],
                "baseline_norms": [],
                "losses": [],
                "fraction_same": []}

    def init_data(self):
        # {"epoch": 0,
        #       "selectivities": [0.1, 0.3...],
        #       "fraction_same": [0.9, 0.7...],
        #       "cos_sims": [[0.1, 0.2, 0.3],... ],    # Per variable, per batch
        #       "baseline_norms": [[1, 2, 3],... ], # Per variable, per batch
        #       "losses": [2.7, 2.3, 2.3], 
        # }
        self.data = self.base_dict(0)
        data_pickle_dir = os.path.join(self.pickle_dir, "biases")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                                 "{}_biases".format(self.pickle_prefix))
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def handle_backward_batch(self, batch):
        average_loss = sum([example.loss.item() for example in batch]) / float(len(batch))
        selectivity = sum([1 for example in batch if example.select]) / float(len(batch))
        baseline_norms = batch[0].baseline_norms
        cos_sims = batch[0].cos_sims
        fraction_same = batch[0].fraction_same
        self.data["losses"].append(average_loss)
        self.data["selectivities"].append(selectivity)
        self.data["cos_sims"].append(cos_sims)
        self.data["baseline_norms"].append(baseline_norms)
        self.data["fraction_same"].append(fraction_same)

    def write(self):
        epoch_file = "{}.epoch_{}.pickle".format(self.data_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % self.log_frequency == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ImageWriter(object):
    def __init__(self, data_dir, dataset, unnormalizer):
        self.data_dir = data_dir
        self.dataset = dataset
        self.unnormalizer = unnormalizer
        self.init_data()

    def init_data(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.output_dir = os.path.join(self.data_dir, "{}_by_id".format(self.dataset))
        print(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def write_partition(self, partition):
        to_pil = torchvision.transforms.ToPILImage()
        for elem in partition:
            img_tensor = elem[0].cpu()
            unnormalized = self.unnormalizer(img_tensor)
            img = to_pil(unnormalized)

            img_id = elem[2]
            img_file = os.path.join(self.output_dir, "image-{}.png".format(img_id))

            img.save(img_file, 'PNG')


class ProbabilityByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.data = {}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "probabilities_by_image")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                             "{}_probabilities".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, probabilities):
        for image_id, probability in zip(image_ids, probabilities):
            if image_id not in self.data.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.data[image_id] = []
            self.data[image_id].append(probability)

    def handle_backward_batch(self, batch):
        ids = [example.image_id.item() for example in batch]
        probabilities = [example.select_probability for example in batch]
        self.update_data(ids, probabilities)

    def write(self):
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ImageIdHistLogger(object):

    def __init__(self, pickle_dir, pickle_prefix, num_images, log_interval):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.log_interval = log_interval
        self.init_data(num_images)

    def next_epoch(self):
        self.write()
        self.current_epoch += 1

    def init_data(self, num_images):
        # Store frequency of each image getting backpropped
        keys = range(num_images)
        self.data = dict(zip(keys, [0] * len(keys)))
        data_pickle_dir = os.path.join(self.pickle_dir, "image_id_hist")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                                 "{}_images_hist".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids):
        for chosen_id in image_ids:
            self.data[chosen_id] += 1

    def handle_backward_batch(self, batch):
        ids = [example.image_id.item() for example in batch if example.select]
        self.update_data(ids)

    def write(self):
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        epoch_file = "{}.epoch_{}.pickle".format(self.data_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % self.log_interval == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class LossesByEpochLogger(object):

    def __init__(self, pickle_dir, pickle_prefix, log_frequency):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.log_frequency = log_frequency
        self.pickle_prefix = pickle_prefix
        self.init_data()

    def next_epoch(self):
        self.write()
        self.current_epoch += 1
        self.data = []

    def init_data(self):
        # Store frequency of each image getting backpropped
        self.data = []
        data_pickle_dir = os.path.join(self.pickle_dir, "losses")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                                 "{}_losses".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, losses):
        self.data += losses

    def handle_backward_batch(self, batch):
        losses = [example.loss.item() for example in batch]
        self.update_data(losses)

    def write(self):
        epoch_file = "{}.epoch_{}.pickle".format(self.data_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % self.log_frequency == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class LossesByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.data = {}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "losses_by_image")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                             "{}_losses".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, losses):
        for image_id, loss in zip(image_ids, losses):
            if image_id not in self.data.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.data[image_id] = []
            self.data[image_id].append(loss)

    def handle_backward_batch(self, batch):
        ids = [example.image_id for example in batch]
        losses = [example.loss for example in batch]
        self.update_data(ids, losses)

    def write(self):
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class VariancesByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.data = {}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "variance_by_image")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                             "{}_variances".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, losses):
        for image_id, loss in zip(image_ids, losses):
            if image_id not in self.data.keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.data[image_id] = []
            self.data[image_id].append(loss)

    def handle_backward_batch(self, batch):
        ids = [example.image_id for example in batch]
        losses = [example.loss for example in batch]
        self.update_data(ids, losses)

    def write(self):
        variance = {}
        for image_id in self.data.keys():
            variance[image_id] = np.var(self.data[image_id])
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(variance, handle, protocol=pickle.HIGHEST_PROTOCOL)

class VariancesByEpochLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, log_frequency):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.log_frequency = log_frequency
        self.pickle_prefix = pickle_prefix
        self.init_data()

    def next_epoch(self):
        self.write()
        self.current_epoch += 1
        self.data = []

    def init_data(self):
        # Store frequency of each image getting backpropped
        self.data = []
        data_pickle_dir = os.path.join(self.pickle_dir, "variance_by_epoch")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                                 "{}_variances".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, variance):
        self.data += [variance]

    def handle_backward_batch(self, batch):
        losses = [example.loss.item() for example in batch]
        variance = np.var(losses)
        self.update_data(variance)

    def write(self):
        epoch_file = "{}.epoch_{}.pickle".format(self.data_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % self.log_frequency == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class VariancesByAverageProbabilityByImageLogger(object):
    def __init__(self, pickle_dir, pickle_prefix, max_num_images=None):
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_data()
        self.max_num_images = max_num_images
        self.data = {"losses": {}, "probabilities": {}}

    def next_epoch(self):
        self.write()

    def init_data(self):
        # Store frequency of each image getting backpropped
        data_pickle_dir = os.path.join(self.pickle_dir, "variance_by_avg_prob")
        self.data_pickle_file = os.path.join(data_pickle_dir,
                                             "{}_variances".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)

    def update_data(self, image_ids, probabilities, losses):
        for image_id, prob, loss in zip(image_ids, probabilities, losses):
            if image_id not in self.data["losses"].keys():
                if self.max_num_images:
                    if image_id >= self.max_num_images:
                        continue
                self.data["losses"][image_id] = []
                self.data["probabilities"][image_id] = []
            self.data["losses"][image_id].append(loss)
            self.data["probabilities"][image_id].append(prob)

    def handle_backward_batch(self, batch):
        ids = [example.image_id for example in batch]
        losses = [example.loss for example in batch]
        probabilities = [example.select_probability for example in batch]
        self.update_data(ids, probabilities, losses)

    def write(self):
        out = {}
        for image_id in self.data["losses"].keys():
            var = np.var(self.data["losses"][image_id])
            avg_prob = np.average(self.data["probabilities"][image_id])
            out[image_id] = (avg_prob, var)
        latest_file = "{}.pickle".format(self.data_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Logger(object):

    def __init__(self, log_interval=1, epoch=0, num_backpropped=0, num_skipped=0):
        self.current_epoch = epoch
        self.current_batch = 0
        self.log_interval = log_interval

        self.global_num_backpropped = num_backpropped
        self.global_num_skipped = num_skipped

        self.partition_loss = 0
        self.partition_backpropped_loss = 0
        self.partition_num_backpropped = 0
        self.partition_num_skipped = 0
        self.partition_num_correct = 0

    def next_epoch(self):
        self.current_epoch += 1

    @property
    def partition_seen(self):
        return self.partition_num_backpropped + self.partition_num_skipped

    @property
    def average_partition_loss(self):
        return self.partition_loss / float(self.partition_seen)

    @property
    def average_partition_backpropped_loss(self):
        return self.partition_backpropped_loss / float(self.partition_num_backpropped)

    @property
    def partition_accuracy(self):
        return 100. * self.partition_num_correct / self.partition_seen

    @property
    def train_debug(self):
        return 'train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
            self.current_epoch,
            self.global_num_backpropped,
            self.global_num_skipped,
            self.average_partition_loss,
            self.average_partition_backpropped_loss,
            time.time(),
            self.partition_accuracy)

    def next_partition(self):
        self.partition_loss = 0
        self.partition_backpropped_loss = 0
        self.partition_num_backpropped = 0
        self.partition_num_skipped = 0
        self.partition_num_correct = 0

    def handle_forward_batch(self, batch):
        # Populate batch_stats
        self.partition_loss += sum([example.loss for example in batch])

    def handle_backward_batch(self, batch):

        self.current_batch += 1

        num_backpropped = sum([int(example.select) for example in batch])
        num_skipped = sum([int(not example.select) for example in batch])
        self.global_num_backpropped += num_backpropped
        self.global_num_skipped += num_skipped

        self.partition_num_backpropped += num_backpropped
        self.partition_num_skipped += num_skipped
        self.partition_backpropped_loss += sum([example.backpropped_loss
                                                for example in batch
                                                if example.backpropped_loss])
        self.partition_num_correct += sum([int(example.is_correct) for example in batch])

        self.write()

    def write(self):
        if self.current_batch % self.log_interval == 0:
            print(self.train_debug)


