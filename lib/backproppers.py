import torch
import torch.nn as nn

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

    def _get_all_probabilities_tensor(self, batch):
        probabilities = [example.select_probability for example in batch]
        return torch.tensor(probabilities, dtype=torch.float)

    def backward_pass(self, batch):
        self.net.train()

        data = self._get_chosen_data_tensor(batch)
        targets = self._get_chosen_targets_tensor(batch)
        #probabilities = self._get_chosen_probabilities_tensor(batch)
        probabilities = self._get_all_probabilities_tensor(batch)

        average_prob = torch.mean(probabilities)
        print("Average prob {}".format(average_prob))

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        outputs = self.net(data) 
        losses = self.loss_fn(reduce=False)(outputs, targets)
        print("losses 1 {}".format(losses))

        # Scale each loss by image-specific select probs
        #losses = torch.div(losses, probabilities.to(self.device))
        losses = torch.mul(losses, probabilities.to(self.device))
        print("losses 2 {}".format(losses))
        losses = torch.div(losses, average_prob.to(self.device))
        print("losses 3 {}".format(losses))


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




