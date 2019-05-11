import math
import numpy as np
import torch
import torch.nn as nn
from timeit import default_timer as timer


def CosineSim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

class PrimedBackpropper(object):
    def __init__(self, initial, final, initial_epochs, epoch=0):
        self.epoch = epoch
        self.initial = initial
        self.final = final
        self.initial_epochs = initial_epochs

    def next_epoch(self):
        self.epoch += 1

    def get_backpropper(self):
        return self.initial if self.epoch < self.initial_epochs else self.final

    @property
    def optimizer(self):
        return self.initial.optimizer if self.epoch < self.initial_epochs else self.final.optimizer

    def backward_pass(self, *args, **kwargs):
        return self.get_backpropper().backward_pass(*args, **kwargs)


class BaselineBackpropper(object):

    def __init__(self, device, net, optimizer, loss_fn):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.loss_fn = loss_fn
        # TODO: This doesn't work after resuming from checkpoint

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [example.datum for example in batch if example.select]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [example.target for example in batch if example.select]
        chosen = len([example.target for example in batch if example.select])
        whole = len(batch)
        return torch.stack(chosen_targets)

    def _get_chosen_probabilities_tensor(self, batch):
        probabilities = [example.select_probability for example in batch if example.select]
        return torch.tensor(probabilities, dtype=torch.float)

    def backward_pass(self, batch):
        self.net.train()

        data = self._get_chosen_data_tensor(batch)
        targets = self._get_chosen_targets_tensor(batch)
        probabilities = self._get_chosen_probabilities_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)

        # Add for logging selected loss
        for example, loss in zip(batch, losses):
            example.backpropped_loss = loss.item()

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return batch


class SamplingBackpropper(object):

    def __init__(self, device, net, optimizer, loss_fn):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.loss_fn = loss_fn

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [example.datum for example in batch if example.select]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [example.target for example in batch if example.select]
        return torch.stack(chosen_targets)

    def _get_chosen_probabilities_tensor(self, batch):
        probabilities = [example.select_probability for example in batch if example.select]
        return torch.tensor(probabilities, dtype=torch.float)

    def backward_pass(self, batch):
        self.net.train()

        data = self._get_chosen_data_tensor(batch)
        targets = self._get_chosen_targets_tensor(batch)
        probabilities = self._get_chosen_probabilities_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)

        # Scale each loss by image-specific select probs
        #losses = torch.div(losses, probabilities.to(self.device))

        # Add for logging selected loss
        for example, loss in zip(batch, losses):
            example.backpropped_loss = loss.item()

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return batch

class GradientAndSelectivityLoggingBackpropper(SamplingBackpropper):

    def __init__(self, device, net, optimizer, loss_fn, selectivity_resolution, epoch_log_interval):
        super(GradientAndSelectivityLoggingBackpropper, self).__init__(device, net, optimizer, loss_fn)
        self.selectivity_resolution = selectivity_resolution
        self.epoch_log_interval = epoch_log_interval
        self.epoch = 0

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

    def next_epoch(self):
        self.epoch += 1

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

        if self.epoch % self.epoch_log_interval == 0:
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
        for example, baseline_loss in zip(batch, baseline_losses):
            example.backpropped_loss = baseline_loss.item()

        return batch

class RandomGradientAndSelectivityLoggingBackpropper(GradientAndSelectivityLoggingBackpropper):

    def __init__(self, device, net, optimizer, loss_fn, selectivity_resolution, epoch_log_interval):
        super(RandomGradientAndSelectivityLoggingBackpropper, self).__init__(device,
                                                                             net,
                                                                             optimizer,
                                                                             loss_fn,
                                                                             selectivity_resolution,
                                                                             epoch_log_interval)

    def _get_data_subset(self, batch, fraction):
        subset_size = int(fraction * len(batch))
        subset = [batch[i] for i in sorted(random.sample(range(len(batch)), subset_size))]
        chosen_losses = [exp.loss for exp in subset]
        return self._get_data_tensor(subset), self._get_targets_tensor(subset)

class ReweightedBackpropper(object):

    def __init__(self, device, net, optimizer, loss_fn):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.loss_fn = loss_fn

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [example.datum for example in batch if example.select]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [example.target for example in batch if example.select]
        return torch.stack(chosen_targets)

    def _get_chosen_probabilities_tensor(self, batch):
        probabilities = [example.select_probability for example in batch if example.select]
        return torch.tensor(probabilities, dtype=torch.float)

    def _get_chosen_weights_tensor(self, batch):
        prob_sum = sum([example.select_probability for example in batch])
        probabilities = [prob_sum / len(batch) / example.select_probability for example in batch]
        return torch.tensor(probabilities, dtype=torch.float)

    def backward_pass(self, batch):
        self.net.train()

        data = self._get_chosen_data_tensor(batch)
        targets = self._get_chosen_targets_tensor(batch)
        weights = self._get_chosen_weights_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)

        # Scale each loss by image-specific select probs
        losses = torch.div(losses, weights.to(self.device))

        # Add for logging selected loss
        for example, loss in zip(batch, losses):
            example.backpropped_loss = loss.item()

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return batch




