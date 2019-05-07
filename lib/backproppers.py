import numpy as np
import torch
import torch.nn as nn

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

class GradientLoggingSamplingBackpropper(SamplingBackpropper):

    def __init__(self, device, net, optimizer, loss_fn):
        super(GradientLoggingSamplingBackpropper, self).__init__(device, net, optimizer, loss_fn)

    def _get_data_tensor(self, batch):
        data = [example.datum for example in batch]
        return torch.stack(data)

    def _get_targets_tensor(self, batch):
        targets = [example.target for example in batch]
        return torch.stack(targets)


    def backward_pass(self, batch):

        self.net.train()

        chosen_data = self._get_chosen_data_tensor(batch)
        chosen_targets = self._get_chosen_targets_tensor(batch)

        data = self._get_data_tensor(batch)
        targets = self._get_targets_tensor(batch)
        probabilities = self._get_chosen_probabilities_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        baseline_outputs = self.net(data) 
        chosen_outputs = self.net(chosen_data) 

        baseline_losses = self.loss_fn(reduce=False)(baseline_outputs, targets)
        chosen_losses = self.loss_fn(reduce=False)(chosen_outputs, chosen_targets)

        # Add for logging selected loss
        for example, baseline_loss, chosen_loss in zip(batch, baseline_losses, chosen_losses):
            example.backpropped_loss = baseline_loss.item()
            example.chosen_loss = chosen_loss.item()

        # Reduce loss
        baseline_loss = baseline_losses.mean()
        chosen_loss = chosen_losses.mean()

        # Calculate chosen gradients
        self.optimizer.zero_grad()
        chosen_loss.backward()

        # Log chosen gradients
        chosen_grads = []
	for p in self.net.parameters():
            chosen_grads.append(p.grad.data.cpu().numpy().flatten())

        # Calculate baseline gradients
        self.optimizer.zero_grad()
        baseline_loss.backward()

        # Log baseline gradients
        cosine_sims = []
        baseline_norms = []
	for p, chosen_grad in zip(self.net.parameters(), chosen_grads):
            baseline_grad = p.grad.data.cpu().numpy().flatten()
            baseline_norms.append(torch.norm(p.grad.data))
            cosine_sim = CosineSim(baseline_grad, chosen_grad)
            cosine_sims.append(cosine_sim)

        # Dirty hack to add logging data to first example :#
        batch[0].cos_sims = cosine_sims
        batch[0].baseline_norms = baseline_norms

        # Update weights with baseline losses
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




