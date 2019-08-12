from scipy import stats
import collections
import math
import numpy as np
import torch
import torch.nn as nn
from random import shuffle

# TODO: Transform into base classes
def get_selector(selector_type, probability_calculator, num_images_to_prime, sample_size, forwards):
    if selector_type == "sampling":
        final_selector = SamplingSelector(probability_calculator, forwards)
    elif selector_type == "alwayson":
        final_selector = AlwaysOnSelector(probability_calculator, forwards)
    elif selector_type == "deterministic":
        final_selector = DeterministicSamplingSelector(probability_calculator, forwards)
    elif selector_type == "baseline":
        final_selector = BaselineSelector()
    elif selector_type == "topk":
        final_selector = TopKSelector(probability_calculator,
                                      sample_size,
                                      forwards)
    elif selector_type == "lowk":
        final_selector = LowKSelector(probability_calculator,
                                      sample_size,
                                      forwards)
    elif selector_type == "randomk":
        final_selector = RandomKSelector(probability_calculator,
                                         sample_size,
                                         forwards)
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
    def __init__(self, probability_calculator, sample_size, forwards=False):
        self.get_select_probability = probability_calculator.get_probability
        self.sample_size = sample_size

    def select(self, example):
        select_probability = example.get_sp(self.forwards)
        if hasattr(example, "fp_draw"):
            draw = example.fp_draw
        else:
            draw = np.random.uniform(0, 1)
        return draw < select_probability.item()

    def get_indices(self, sps):
        indices = np.array(sps).argsort()[-self.sample_size:]
        return indices

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            example.set_sp(self.get_select_probability(example), self.forwards)
        sps = [example.get_sp(self.forwards) for example in forward_pass_batch]
        indices = self.get_indices(sps) 
        for i in range(len(forward_pass_batch)):
            if i in indices:
                forward_pass_batch[i].set_select(True, self.forwards)
            else:
                forward_pass_batch[i].set_select(False, self.forwards)
        return forward_pass_batch


class LowKSelector(TopKSelector):
    def __init__(self, probability_calculator, sample_size, forwards=False):
        super(LowKSelector, self).__init__(probability_calculator,
                                           sample_size,
                                           forwards)

    def get_indices(self, sps):
        indices = np.array(sps).argsort()[:self.sample_size]
        return indices


class RandomKSelector(TopKSelector):
    def __init__(self, probability_calculator, sample_size, forwards=False):
        super(RandomKSelector, self).__init__(probability_calculator,
                                              sample_size,
                                              forwards)

    def get_indices(self, sps):
        all_indices = np.array(sps).argsort()
        shuffle(all_indices)
        indices = all_indices[:self.sample_size]
        return indices


class SamplingSelector(object):
    def __init__(self, probability_calculator, forwards=False):
        self.get_select_probability = probability_calculator.get_probability
        self.forwards = forwards

    def select(self, example):
        select_probability = example.get_sp(self.forwards)
        if hasattr(example, "fp_draw"):
            draw = example.fp_draw
            print("Use old fp_draw: {:2f} > {:2f}".format(draw,
                                                            select_probability))
        else:
            draw = np.random.uniform(0, 1)
        return draw < select_probability

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            prob = self.get_select_probability(example)
            example.set_sp(prob, self.forwards)
            is_selected = self.select(example)
            example.set_select(is_selected, self.forwards)
        return forward_pass_batch


class AlwaysOnSelector(SamplingSelector):
    def __init__(self, probability_calculator, forwards=False):
        super(AlwaysOnSelector, self).__init__(probability_calculator, forwards)

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
        select_probability = example.get_sp(self.forwards)
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
            example.set_sp(sp_tensor.item(), self.forwards)
            self.increase_select_sum(example)
            example.set_select(self.select(example), self.forwards)
            if example.get_select(self.forwards):
                self.decrease_select_sum(example)
        return forward_pass_batch

class BaselineSelector(object):

    def __init__(self, forwards=False):
        self.forwards = forwards

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            example.set_sp(torch.tensor([[1]]).item(), self.forwards)
            example.set_select(True, self.forwards)
        return forward_pass_batch


