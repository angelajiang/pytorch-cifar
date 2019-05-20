from scipy import stats
import collections
import math
import numpy as np
import torch
import torch.nn as nn
from random import shuffle

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


class LowKSelector(object):
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
        indices = np.array(sps).argsort()[:self.sample_size]

        for i in range(len(forward_pass_batch)):
            if i in indices:
                forward_pass_batch[i].select = True
            else:
                forward_pass_batch[i].select = False
        return forward_pass_batch

class RandomKSelector(object):
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
        all_indices = np.array(sps).argsort()
        shuffle(all_indices)
        indices = all_indices[:self.sample_size]

        for i in range(len(forward_pass_batch)):
            if i in indices:
                forward_pass_batch[i].select = True
            else:
                forward_pass_batch[i].select = False
        return forward_pass_batch


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

class DeterministicSamplingSelector(object):
    def __init__(self, probability_calculator, initial_sum=0):
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


class RelativeProbabilityCalculator(object):
    def __init__(self, device, loss_fn, sampling_min, history_length):
        self.device = device
        self.loss_fn = loss_fn
        self.historical_losses = collections.deque(maxlen=history_length)
        self.sampling_min = sampling_min

    def update_history(self, loss):
        self.historical_losses.append(loss)

    def calculate_probability(self, percentile):
        return percentile / 100.

    def get_probability(self, example):
        loss = self.loss_fn()(example.output.unsqueeze(0), example.target.unsqueeze(0))
        loss = loss.cpu().data.numpy()
        self.update_history(loss)
        prob = self.calculate_probability(stats.percentileofscore(self.historical_losses, loss, kind="rank"))
        return max(self.sampling_min, prob)

class RelativeSquaredProbabilityCalculator(RelativeProbabilityCalculator):
    def __init__(self, device, loss_fn, sampling_min, history_length):
        super(RelativeSquaredProbabilityCalculator, self).__init__(device, loss_fn, sampling_min, history_length)

    def calculate_probability(self, percentile):
        return math.pow(percentile / 100., 2)

class RelativeCubedProbabilityCalculator(RelativeProbabilityCalculator):
    def __init__(self, device, loss_fn, sampling_min, history_length):
        super(RelativeCubedProbabilityCalculator, self).__init__(device, loss_fn, sampling_min, history_length)

    def calculate_probability(self, percentile):
        return math.pow(percentile / 100., 3)

class RelativeSeventhProbabilityCalculator(RelativeProbabilityCalculator):
    def __init__(self, device, loss_fn, sampling_min, history_length):
        super(RelativeSeventhProbabilityCalculator, self).__init__(device, loss_fn, sampling_min, history_length)

    def calculate_probability(self, percentile):
        return math.pow(percentile / 100., 7)

class SelectProbabiltyCalculator(object):
    def __init__(self, sampling_min, sampling_max, num_classes, device,
                 square=False, prob_transform=None):
        # prob_transform should be a function f where f(x) <= 1
        self.sampling_min = sampling_min
        self.sampling_max = sampling_max
        self.num_classes = num_classes
        self.device = device
        self.square = square
        if prob_transform:
            self.prob_transform = prob_transform
        else:
            self.prob_transform  = lambda x: x

    def get_probability(self, example):
        target = example.target
        softmax_output = example.softmax_output
        target_tensor = self.hot_encode_scalar(target)
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        if self.square:
            l2_dist *= l2_dist
        base = torch.clamp(self.prob_transform(l2_dist), min=self.sampling_min)
        prob = torch.clamp(base, max=self.sampling_max).detach()
        return prob.item()

    def hot_encode_scalar(self, target):
        target_vector = np.zeros(self.num_classes)
        target_vector[target.item()] = 1
        target_tensor = torch.Tensor(target_vector)
        return target_tensor



class PScaledProbabiltyCalculator(object):
    def __init__(self,
                 sampling_min,
                 sampling_max,
                 num_classes,
                 device,
                 update_steps,
                 square=False,
                 prob_transform=None):
        self.sampling_min = sampling_min
        self.sampling_max = sampling_max
        self.num_classes = num_classes
        self.device = device
        self.square = square
        if prob_transform:
            self.prob_transform = prob_transform
        else:
            self.prob_transform  = lambda x: x

        # Scale probabilities so hardest examples are at p = 1
        self.update_steps = update_steps
        self.current_step = 0
        self.current_max_prob = 1
        self.next_max_prob = 0

    def update_pscale(self, p):
        self.current_step += 1

        if p > self.current_max_prob:
            # Update right away so your scalar never
            # gives you a probability higher than 1
            self.current_max_prob = p
            print("update_pscale 0, {}".format(self.pscale))

        if p > self.next_max_prob:
            self.next_max_prob = p

        if self.current_step == self.update_steps:
            self.current_max_prob = self.next_max_prob
            print("update_pscale 1, {}".format(self.pscale))
            self.current_step = 0
            self.next_max_prob = 0

    @property
    def pscale(self):
        return 1. / self.current_max_prob

    def get_probability(self, example):
        target = example.target
        softmax_output = example.softmax_output
        target_tensor = self.hot_encode_scalar(target)
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        if self.square:
            l2_dist *= l2_dist
        p = torch.clamp(self.prob_transform(l2_dist),
                        min=self.sampling_min,
                        max=self.sampling_max).detach()
        self.update_pscale(p)
        pscaled_p = p * self.pscale
        return pscaled_p.item()

    def hot_encode_scalar(self, target):
        target_vector = np.zeros(self.num_classes)
        target_vector[target.item()] = 1
        target_tensor = torch.Tensor(target_vector)
        return target_tensor


class ProportionalProbabiltyCalculator(object):
    def __init__(self, sampling_min, sampling_max, num_classes, device,
                 square=False, prob_transform=None):
        self.sampling_min = sampling_min
        self.sampling_max = sampling_max
        self.num_classes = num_classes
        self.device = device
        self.square = square

        if self.square:
            self.theoretical_max = 2
        else:
            self.theoretical_max = math.sqrt(2)

        # prob_transform should be a function f where f(x) <= 1
        if prob_transform:
            self.prob_transform = prob_transform
        else:
            self.prob_transform  = lambda x: x

    def get_probability(self, example):
        target = example.target
        softmax_output = example.softmax_output
        target_tensor = self.hot_encode_scalar(target)
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        if self.square:
            l2_dist *= l2_dist
        prob = l2_dist / float(self.theoretical_max)
        transformed_prob = self.prob_transform(prob)
        clamped_prob = torch.clamp(transformed_prob,
                                   min=self.sampling_min)
        return clamped_prob.item()

    def hot_encode_scalar(self, target):
        target_vector = np.zeros(self.num_classes)
        target_vector[target.item()] = 1
        target_tensor = torch.Tensor(target_vector)
        return target_tensor
