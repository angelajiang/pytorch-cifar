import torch
import time
import torch.nn as nn

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

        data = self._get_chosen_data_tensor(batch).to(self.device)
        targets = self._get_chosen_targets_tensor(batch).to(self.device)

        probabilities = self._get_chosen_probabilities_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        outputs = self.net(data) 
        loss = self.loss_fn(reduce=True)(outputs, targets)

	softmax_outputs = nn.Softmax()(outputs)
        _, predicted = outputs.max(1)
        is_corrects = predicted.eq(targets)

        # Add for logging selected loss
        for example, output, softmax_output, is_correct in zip(batch,
                                                               outputs,
                                                               softmax_outputs,
                                                               is_corrects):
            example.output = output.detach().cpu()
            example.softmax_output = softmax_output.detach().cpu()
            example.correct = is_correct.item()

        # Reduce loss
        # loss = losses.mean()

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

    def backward_pass(self, batch):
        self.net.train()

        data = self._get_chosen_data_tensor(batch).to(self.device)
        targets = self._get_chosen_targets_tensor(batch).to(self.device)

        # Run forward pass
        outputs = self.net(data) 
        loss = self.loss_fn(reduce=True)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)             # OPT: not necessary when logging is off
        _, predicted = outputs.max(1)
        is_corrects = predicted.eq(targets)

        # Scale each loss by image-specific select probs
        #losses = torch.div(losses, probabilities.to(self.device))

        # Add for logging selected loss
        for example, output, softmax_output, is_correct in zip(batch,
                                                               outputs,
                                                               softmax_outputs,
                                                               is_corrects):
            example.output = output.detach().cpu()
            example.softmax_output = softmax_output.detach().cpu()
            example.correct = is_correct.item()

        # Reduce loss
        # loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return batch


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

    @property
    def total_norm(self):
        total_norm = 0
	for p in self.net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

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




