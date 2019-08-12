from scipy import stats
import collections
import math
import numpy as np
import torch
import torch.nn as nn
from random import shuffle
import lib.predictors

# TODO: Transform into base classes
def get_probability_calculator(calculator_type,
                               device,
                               prob_loss_fn,
                               sampling_min,
                               sampling_max,
                               num_classes,
                               max_history_len,
                               prob_pow):
    ## Setup Trainer:ProbabilityCalculator ##
    if prob_pow:
        prob_transform = lambda x: torch.pow(x, prob_pow)
    else:
        prob_transform = None

    if calculator_type == "vanilla":
        probability_calculator = SelectProbabiltyCalculator(sampling_min,
                                                            sampling_max,
                                                            num_classes,
                                                            device,
                                                            prob_transform=prob_transform)
    elif calculator_type == "relative":
        probability_calculator = RelativeProbabilityCalculator(device,
                                                               prob_loss_fn,
                                                               sampling_min,
                                                               max_history_len,
                                                               prob_pow)
    elif calculator_type == "hybrid":
        probability_calculator = HybridProbabilityCalculator(device,
                                                             prob_loss_fn,
                                                             sampling_min,
                                                             max_history_len,
                                                             prob_pow,
                                                             num_classes)
    elif calculator_type == "proportional":
        probability_calculator = ProportionalProbabiltyCalculator(sampling_min,
                                                                  sampling_max,
                                                                  num_classes,
                                                                  device,
                                                                  prob_transform=prob_transform)
    else:
        print("Use prob-strategy in {vanilla, relative, hybrid, pscale, proportional}")
        exit()
    return probability_calculator

class RelativeProbabilityCalculator(object):
    def __init__(self, device, loss_fn, sampling_min, history_length, beta):
        self.device = device
        self.loss_fn = loss_fn
        self.historical_losses = collections.deque(maxlen=history_length)
        self.sampling_min = sampling_min
        self.beta = beta

    def update_history(self, loss):
        self.historical_losses.append(loss)

    def calculate_probability(self, percentile):
        return math.pow(percentile / 100., self.beta)

    def get_probability(self, example):
        loss = self.loss_fn()(example.output.unsqueeze(0), example.target.unsqueeze(0))
        loss = loss.cpu().data.numpy()
        self.update_history(loss)
        prob = self.calculate_probability(stats.percentileofscore(self.historical_losses, loss, kind="rank"))
        return max(self.sampling_min, prob)

class HybridProbabilityCalculator(RelativeProbabilityCalculator):
    def __init__(self, device, loss_fn, sampling_min, history_length, beta, num_classes):
        RelativeProbabilityCalculator.__init__(self, device, loss_fn, sampling_min, history_length, beta)
        self.num_classes = num_classes

    def get_relative_probability(self, example):
        loss = self.loss_fn()(example.output.unsqueeze(0), example.target.unsqueeze(0))
        loss = loss.cpu().data.numpy()
        self.update_history(loss)
        prob = self.calculate_probability(stats.percentileofscore(self.historical_losses, loss, kind="rank"))
        return max(self.sampling_min, prob)

    def get_absolute_probability(self, example):
        target = example.target
        softmax_output = example.softmax_output
        target_tensor = example.hot_encoded_target
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        l2_dist *= l2_dist
        base = torch.clamp(l2_dist, min=self.sampling_min)
        prob = torch.clamp(base, max=1).detach()
        return prob.item()

    def get_probability(self, example):
        return max(self.get_relative_probability(example), self.get_absolute_probability(example))

class SelectProbabiltyCalculator(object):
    def __init__(self, sampling_min, sampling_max, num_classes, device,
                 prob_transform=None):
        # prob_transform should be a function f where f(x) <= 1
        self.sampling_min = sampling_min
        self.sampling_max = sampling_max
        self.num_classes = num_classes
        self.device = device
        if prob_transform:
            self.prob_transform = prob_transform
        else:
            self.prob_transform  = lambda x: x

    def get_probability(self, example):
        target = example.target
        softmax_output = example.softmax_output
        target_tensor = example.hot_encoded_target
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        l2_dist *= l2_dist
        base = torch.clamp(self.prob_transform(l2_dist), min=self.sampling_min)
        prob = torch.clamp(base, max=self.sampling_max).detach()
        return prob.item()

class ProportionalProbabiltyCalculator(object):
    def __init__(self, sampling_min, sampling_max, num_classes, device,
                 prob_transform=None):
        self.sampling_min = sampling_min
        self.sampling_max = sampling_max
        self.num_classes = num_classes
        self.device = device

        self.theoretical_max = 2

        # prob_transform should be a function f where f(x) <= 1
        if prob_transform:
            self.prob_transform = prob_transform
        else:
            self.prob_transform  = lambda x: x

    def get_probability(self, example):
        target = example.target
        softmax_output = example.softmax_output
        target_tensor = example.hot_encoded_target
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        l2_dist *= l2_dist
        prob = l2_dist / float(self.theoretical_max)
        transformed_prob = self.prob_transform(prob)
        clamped_prob = torch.clamp(transformed_prob,
                                   min=self.sampling_min)
        return clamped_prob.item()

class HistoricalProbabilityCalculator(object):
    def __init__(self, calculator_type, std_multiplier=None, bp_probability_calculator=None):
        self.type = calculator_type
        if self.type == "vanilla":
            self.calculator = VanillaHistoricalCalculator()
        elif self.type == "mean":
            self.calculator = MeanHistoricalCalculator()
        elif self.type == "gp":
            self.calculator = GPHistoricalCalculator(std_multiplier,
                                                     bp_probability_calculator)
        elif self.type == "rto":
            self.calculator = RTOHistoricalCalculator(std_multiplier,
                                                      bp_probability_calculator)

    def get_probability(self, example):
        return self.calculator.get_probability(example)

class VanillaHistoricalCalculator(object):
    def get_probability(self, example):
        return 1

class MeanHistoricalCalculator(VanillaHistoricalCalculator):
    def __init__(self):
        self.history = {}
        self.history_length = 5

    def update_history(self, example):
        if example.image_id not in self.history.keys():
            self.history[example.image_id] = collections.deque(maxlen=self.history_length)
            # First example won't have a loss yet
        else:
            previous_sp = example.get_sp(False)
            if previous_sp:
                if len(self.history[example.image_id]) == 0:
                    self.history[example.image_id].append(previous_sp)
                else:
                    # Check that loss has been updated
                    if previous_sp != self.history[example.image_id][-1]:
                        self.history[example.image_id].append(previous_sp)

    def get_probability(self, example):
        self.update_history(example)
        hist = self.history[example.image_id]
        if not example.get_select(True):
            return 1
        if len(hist) >= self.history_length:
            #print(hist)
            if all(h < 0.001 for h in hist):
                return 0
        return 1

class GPHistoricalCalculator(VanillaHistoricalCalculator):
    def __init__(self, std_multiplier, bp_selector):
        self.history = {}
        self.gps = {}
        self.xs = {}
        self.std_multiplier = std_multiplier
        self.bp_selector = bp_selector
        self.min_history = 0
        self.timeout_multiplier = 1
        self.retrain_every = 10

    def update_history(self, example):
        if example.image_id not in self.history.keys():
            # First example won't have a loss yet
            self.history[example.image_id] = []
            predictor = lib.predictors.GPPredictor()
            self.gps[example.image_id] = predictor
            self.xs[example.image_id] = 0
        else:
            previous_sp = example.get_sp(False)
            if len(self.history[example.image_id]) == 0 or previous_sp != self.history[example.image_id][-1]:
                self.history[example.image_id].append(previous_sp)
                ys = self.history[example.image_id]
                if len(ys) >= self.min_history and len(ys) % self.retrain_every == 0:
                    predictor = self.gps[example.image_id]
                    X = np.array(range(len(ys))).reshape(-1, 1)
                    predictor.update(X, ys)
        self.xs[example.image_id] += 1


    def select(self, y, std):
        draw = np.random.uniform(0, 1)
        if self.timeout_multiplier * (y + (self.std_multiplier*std)) > draw:
            self.timeout_multiplier += 10
            return 1, draw
        else:
            self.timeout_multiplier = 1
            return 0, draw

    def get_probability(self, example):
        self.update_history(example)
        predictor = self.gps[example.image_id]
        hist = self.history[example.image_id]
        if len(hist) < self.min_history:
            return 1

        x = self.xs[example.image_id]
        X = np.array([x]).reshape(-1, 1)
        y, std = predictor.predict(x)
        is_selected, draw = self.select(y, std)
        example.fp_draw = draw
        if is_selected == 0:
            if hasattr(example, "loss"):
                self.bp_selector.update_history(example.loss)
        return is_selected

    '''
    def get_probability(self, example):
        self.update_history(example)
        predictor = self.gps[example.image_id]
        hist = self.history[example.image_id]
        if not example.get_select(True):
            return 1
        if len(hist) >= self.min_history:
            x = self.xs[example.image_id]
            y, _ = predictor.predict(x)
            return y
        return 1
    '''


class RTOHistoricalCalculator(VanillaHistoricalCalculator):
    def __init__(self, std_multiplier, bp_selector):
        self.history = {}
        self.gps = {}
        self.xs = {}
        self.std_multiplier = std_multiplier
        self.bp_selector = bp_selector
        self.min_history = 0
        self.timeout_multiplier = 1
        print("std_multiplier {}".format(self.std_multiplier))

    def update_history(self, example):
        if example.image_id not in self.history.keys():
            # First example won't have a loss yet
            self.history[example.image_id] = []
            predictor = lib.predictors.RTOPredictor()
            self.gps[example.image_id] = predictor
        else:
            predictor = self.gps[example.image_id]
            previous_sp = example.get_sp(False)
            if len(self.history[example.image_id]) == 0 or previous_sp != self.history[example.image_id][-1]:
                self.history[example.image_id].append(previous_sp)
                predictor.update(None, previous_sp)

    def select(self, y, std):
        draw = np.random.uniform(0, 1)
        if self.timeout_multiplier * (y + (self.std_multiplier*std)) > draw:
            self.timeout_multiplier += 2
            return 1, draw
        else:
            self.timeout_multiplier = 1
            return 0, draw

    def get_probability(self, example):
        self.update_history(example)
        predictor = self.gps[example.image_id]
        hist = self.history[example.image_id]
        if len(self.history[example.image_id]) < self.min_history:
            return 1

        y, std = predictor.predict(None)
        is_selected, draw =  self.select(y, std)
        example.fp_draw = draw
        if is_selected == 0:
            if hasattr(example, "loss"):
                self.bp_selector.update_history(example.loss.item())
        return is_selected


