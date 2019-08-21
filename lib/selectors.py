from scipy import stats
import collections
import math
import numpy as np
import torch
import torch.nn as nn
from random import shuffle

# TODO: Transform into base classes
def get_selector(selector_type, probability_calculator, num_images_to_prime, sample_size):
    if selector_type == "sampling":
        final_selector = SamplingSelector(probability_calculator)
    elif selector_type == "alwayson":
        final_selector = AlwaysOnSelector(probability_calculator)
    elif selector_type == "deterministic":
        final_selector = DeterministicSamplingSelector(probability_calculator)
    elif selector_type == "baseline":
        final_selector = BaselineSelector()
    elif selector_type == "topk":
        final_selector = TopKSelector(probability_calculator,
                                      sample_size)
    elif selector_type == "lowk":
        final_selector = LowKSelector(probability_calculator,
                                      sample_size)
    elif selector_type == "randomk":
        final_selector = RandomKSelector(probability_calculator,
                                         sample_size)
    else:
        print("Use sb-strategy in {sampling, deterministic, baseline, topk, lowk, randomk}")
        exit()
    selector = PrimedSelector(BaselineSelector(),
                              final_selector,
                              num_images_to_prime)
    return selector


class PrimedSelector(object):
    def __init__(self, initial, final, initial_num_images, epoch=0):
        self.initial = initial
        self.final = final
        self.initial_num_images = initial_num_images
        self.num_trained = 0

    def next_partition(self, partition_size):
        self.num_trained += partition_size

    def get_selector(self):
        return self.initial if self.num_trained < self.initial_num_images else self.final

    def select(self, *args, **kwargs):
        return self.get_selector().select(*args, **kwargs)

    def mark(self, *args, **kwargs):
        return self.get_selector().mark(*args, **kwargs)


class TopKSelector(object):
    def __init__(self, probability_calculator, sample_size):
        self.get_select_probability = probability_calculator.get_probability
        self.sample_size = sample_size

    def select(self, example):
        select_probability = example.select_probability
        draw = np.random.uniform(0, 1)
        return draw < select_probability.item()

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            example.select_probability = self.get_select_probability(example)
        sps = [example.select_probability for example in forward_pass_batch]
        indices = np.array(sps).argsort()[-self.sample_size:]
        for i in range(len(forward_pass_batch)):
            if i in indices:
                forward_pass_batch[i].select = True
            else:
                forward_pass_batch[i].select = False
        return forward_pass_batch


class LowKSelector(TopKSelector):
    def __init__(self, probability_calculator, sample_size, forwards=False):
        super(LowKSelector, self).__init__(probability_calculator,
                                           sample_size)

    def get_indices(self, sps):
        indices = np.array(sps).argsort()[:self.sample_size]
        return indices


class RandomKSelector(TopKSelector):
    def __init__(self, probability_calculator, sample_size):
        super(RandomKSelector, self).__init__(probability_calculator,
                                              sample_size)

    def get_indices(self, sps):
        all_indices = np.array(sps).argsort()
        shuffle(all_indices)
        indices = all_indices[:self.sample_size]
        return indices


class SamplingSelector(object):
    def __init__(self, probability_calculator):
        self.get_select_probability = probability_calculator.get_probability

    def select(self, example):
        select_probability = example.select_probability
        draw = np.random.uniform(0, 1)
        return draw < select_probability

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            prob = self.get_select_probability(example)
            example.select_probability = prob
            example.select = self.select(example)
        return forward_pass_batch


class AlwaysOnSelector(SamplingSelector):
    def __init__(self, probability_calculator):
        super(AlwaysOnSelector, self).__init__(probability_calculator)

    def select(self, example):
        return True


class DeterministicSamplingSelector(object):
    def __init__(self, probability_calculator, forwards=False, initial_sum=1):
        self.forwards = forwards
        self.global_select_sums = {}
        self.image_ids = set()
        self.get_select_probability = probability_calculator.get_probability
        self.initial_sum = initial_sum

    def increase_select_sum(self, example):
        select_probability = example.select_probability
        image_id = example.image_id.item()
        if image_id not in self.image_ids:
            self.image_ids.add(image_id)
            self.global_select_sums[image_id] = self.initial_sum
        self.global_select_sums[image_id] += select_probability

    def decrease_select_sum(self, example):
        image_id = example.image_id.item()
        self.global_select_sums[image_id] -= 1
        assert(self.global_select_sums[image_id] >= 0)

    def select(self, example):
        image_id = example.image_id.item()
        return self.global_select_sums[image_id] >= 1

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            sp_tensor = self.get_select_probability(example)
            example.select_probability = sp_tensor.item()
            self.increase_select_sum(example)
            example.select = self.select(example)
            if example.select:
                self.decrease_select_sum(example)
        return forward_pass_batch

class BaselineSelector(object):

    def select(self, example):
        return True

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            example.select_probability = torch.tensor([[1]]).item()
            example.select = self.select(example)
        return forward_pass_batch


