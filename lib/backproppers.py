import math
import numpy as np
import torch
import time
import torch.nn as nn
from timeit import default_timer as timer


def CosineSim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

class PrimedBackpropper(object):
    def __init__(self, initial, final, initial_num_images):
        self.initial = initial
        self.final = final
        self.initial_num_images = initial_num_images
        self.num_trained = 0

    def next_partition(self, partition_size):
        self.num_trained += partition_size

    def get_backpropper(self):
        return self.initial if self.num_trained < self.initial_num_images else self.final

    @property
    def optimizer(self):
        return self.initial.optimizer if self.num_trained < self.initial_num_images else self.final.optimizer

    def backward_pass(self, *args, **kwargs):
        return self.get_backpropper().backward_pass(*args, **kwargs)


class SamplingBackpropper(object):

    def __init__(self, device, net, optimizer, loss_fn):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.loss_fn = loss_fn

    def _get_chosen_examples(self, batch):
        return [em for em in batch if em.example.select]

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [em.example.datum for em in batch]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [em.example.target for em in batch]
        return torch.stack(chosen_targets)

    def backward_pass(self, batch):
        self.net.train()

        chosen_batch = self._get_chosen_examples(batch)
        data = self._get_chosen_data_tensor(chosen_batch).to(self.device)
        targets = self._get_chosen_targets_tensor(chosen_batch).to(self.device)

        # Run forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)             # OPT: not necessary when logging is off
        _, predicted = outputs.max(1)
        is_corrects = predicted.eq(targets)

        # Scale each loss by image-specific select probs
        #losses = torch.div(losses, probabilities.to(self.device))

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return batch

class GradientAndSelectivityLoggingBackpropper(SamplingBackpropper):

    def __init__(self, device, net, optimizer, loss_fn, selectivity_resolution, examples_log_interval):
        super(GradientAndSelectivityLoggingBackpropper, self).__init__(device, net, optimizer, loss_fn)
        self.selectivity_resolution = selectivity_resolution
        self.examples_log_interval = examples_log_interval
        self.num_trained = 0

    def _get_data_tensor(self, batch):
        data = [example.datum for example in batch]
        return torch.stack(data)

    def _get_targets_tensor(self, batch):
        targets = [example.target for example in batch]
        return torch.stack(targets)

    def _get_data_subset(self, batch, fraction):
        subset_size = int(fraction * len(batch))
        subset = sorted(batch, key=lambda x: x.loss, reverse=True)[:subset_size]
        chosen_losses = [exp.loss for exp in subset]
        return self._get_data_tensor(subset), self._get_targets_tensor(subset)

    def log_gradients(self, batch):

        baseline_data, baseline_targets = self._get_data_subset(batch, 1)
        baseline_outputs = self.net(baseline_data) 
        baseline_loss = self.loss_fn(reduce=True)(baseline_outputs, baseline_targets)

        self.optimizer.zero_grad()
        baseline_loss.backward(retain_graph=True)

        baseline_grads = []
        for p in self.net.parameters():
            baseline_grads.append(p.grad.data.cpu().numpy().flatten())

        fractions = np.arange(0.1, 1 + 1. / self.selectivity_resolution, 1. / self.selectivity_resolution)
        cosine_sim_data = {}
        fraction_same_data = {}
        for fraction in fractions:
            chosen_data, chosen_targets = self._get_data_subset(batch, fraction)
            chosen_outputs = self.net(chosen_data) 
            chosen_loss = self.loss_fn(reduce=True)(chosen_outputs, chosen_targets)

            self.optimizer.zero_grad()
            chosen_loss.backward(retain_graph=True)

            total_same = 0
            total_count = 0
            cosine_sims = []
            for p, baseline_grad in zip(self.net.parameters(), baseline_grads):
                chosen_grad = p.grad.data.cpu().numpy().flatten()
                cosine_sim = CosineSim(baseline_grad, chosen_grad)
                if not math.isnan(cosine_sim):
                    cosine_sims.append(cosine_sim)

                # Keep track of sign changes
                eps = 1e-5
                baseline_grad[np.abs(baseline_grad) < eps] = 0
                chosen_grad[np.abs(chosen_grad) < eps] = 0

                a = np.sign(baseline_grad)
                b = np.sign(chosen_grad)
                ands = a * b

                num_same = np.where(ands >=  0)[0].size
                total_same += num_same
                total_count += len(ands)

            fraction_same = total_same / float(total_count)
            average_cosine_sim = np.average(cosine_sims)
            cosine_sim_data[fraction] = average_cosine_sim
            fraction_same_data[fraction] = fraction_same


        # Dirty hack to add logging data to first example :#
        batch[0].cos_sims = cosine_sim_data
        batch[0].fraction_same = fraction_same_data

        return baseline_loss

    def backward_pass(self, batch):

        self.net.train()

        if self.num_trained % self.examples_log_interval == 0:
            self.log_gradients(batch)

        baseline_data, baseline_targets = self._get_data_subset(batch, 1)
        baseline_outputs = self.net(baseline_data) 
        baseline_losses = self.loss_fn(reduce=False)(baseline_outputs, baseline_targets)
        baseline_loss = baseline_losses.mean()

        # Do an extra backwards pass to make sure we're backpropping baseline
        self.optimizer.zero_grad()
        baseline_loss.backward()
        self.optimizer.step()

        # Add for logging selected loss
        for em, loss, is_correct in zip(chosen_batch,
                                        losses,
                                        is_corrects):
            em.example.loss = loss.item()
            em.example.correct = is_correct.item()
            em.metadata["loss"] = em.example.loss

        return batch

class RandomGradientAndSelectivityLoggingBackpropper(GradientAndSelectivityLoggingBackpropper):

    def __init__(self, device, net, optimizer, loss_fn, selectivity_resolution, examples_log_interval):
        super(RandomGradientAndSelectivityLoggingBackpropper, self).__init__(device,
                                                                             net,
                                                                             optimizer,
                                                                             loss_fn,
                                                                             selectivity_resolution,
                                                                             examples_log_interval)

    def _get_data_subset(self, batch, fraction):
        subset_size = int(fraction * len(batch))
        subset = [batch[i] for i in sorted(random.sample(range(len(batch)), subset_size))]
        chosen_losses = [exp.loss for exp in subset]
        return self._get_data_tensor(subset), self._get_targets_tensor(subset)

class ReweightedBackpropper(SamplingBackpropper):

    def __init__(self, device, net, optimizer, loss_fn):
        super(ReweightedBackpropper, self).__init__(device,
                                                    net,
                                                    optimizer,
                                                    loss_fn)

    def _get_chosen_weights_tensor(self, batch):
        probabilities = [prob_sum / len(batch) / example.select_probability for example in batch]
        return torch.tensor(probabilities, dtype=torch.float)

    def backward_pass(self, batch):
        self.net.train()

        chosen_batch = self._get_chosen_examples(batch)
        data = self._get_chosen_data_tensor(chosen_batch).to(self.device)
        targets = self._get_chosen_targets_tensor(chosen_batch).to(self.device)
        weights = self._get_chosen_weights_tensor(chosen_batch).to(self.device)

        # Run forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)             # OPT: not necessary when logging is off
        _, predicted = outputs.max(1)
        is_corrects = predicted.eq(targets)

        # Scale each loss by image-specific select probs
        losses = torch.mul(losses, weights)

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Add for logging selected loss
        for em, loss, is_correct in zip(chosen_batch,
                                        losses,
                                        is_corrects):
            em.example.loss = loss.item()
            em.example.correct = is_correct.item()
            em.metadata["loss"] = em.example.loss

        return batch

class AlwaysOnBackpropper(object):

    def __init__(self, device, net, optimizer, loss_fn):
        super(SamplingBackpropper, self).__init__(device,
                                                  net,
                                                  optimizer,
                                                  loss_fn)

    def _get_chosen_examples(self, batch):
        return batch

