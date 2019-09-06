import torch.nn.functional as F
import torch

def CrossEntropySquaredLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = -torch.mean(outputs)
            return cross_entropy_loss ** 2 / 10.
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = - outputs
            return cross_entropy_loss ** 2 / 10.
    return fn

def CrossEntropyRegulatedLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            max_others = None
            for output, label in zip(outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other = torch.max(other_outputs)
                if max_others is None:
                    max_others = max_other.unsqueeze(-1)
                else:
                    max_others = torch.cat((max_others, max_other.unsqueeze(-1)))
            cross_entropy_loss = -torch.mean(class_outputs + 0.05 * max_others)
            return cross_entropy_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            max_others = None
            for output, label in zip(outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other = torch.max(other_outputs)
                if max_others is None:
                    max_others = max_other.unsqueeze(-1)
                else:
                    max_others = torch.cat((max_others, max_other.unsqueeze(-1)))
            cross_entropy_loss = -(class_outputs + 0.05 * max_others)
            return cross_entropy_loss
    return fn

def CrossEntropyRegulatedBoostedLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            num_classes = outputs.size()[1]            # num classes
            outputs = F.softmax(outputs, dim=1)       # compute the softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            losses = None
            for output, class_prob, label in zip(outputs, class_outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other_prob = torch.max(other_outputs)
                loss = - torch.log(class_prob) - torch.log(1 - (max_other_prob - (1 - class_prob) / (num_classes - 1)))
                cross_entropy = torch.log(class_prob)
                if losses is None:
                    losses = loss.unsqueeze(-1)
                else:
                    losses = torch.cat((losses, loss.unsqueeze(-1)))
            reduced_loss = torch.mean(losses)
            return reduced_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            num_classes = outputs.size()[1]            # num classes
            outputs = F.softmax(outputs, dim=1)       # compute the softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            losses = None
            for output, class_prob, label in zip(outputs, class_outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other_prob = torch.max(other_outputs)
                loss = - torch.log(class_prob) - torch.log(1 - (max_other_prob - (1 - class_prob) / (num_classes - 1)))
                if losses is None:
                    losses = loss.unsqueeze(-1)
                else:
                    losses = torch.cat((losses, loss.unsqueeze(-1)))
            return losses
    return fn

def CrossEntropyLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = -torch.mean(outputs)
            return cross_entropy_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = - outputs
            return cross_entropy_loss
    return fn

def MSELoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]
            num_classes = outputs.size()[1]
            outputs = F.softmax(outputs, dim=1)
            targets = torch.eye(num_classes)
            l2_dists = None
            for output, label in zip(outputs, labels):
                target = targets[int(label.item())]
                l2_dist = torch.dist(target, output.cpu())
                if l2_dists is None:
                    l2_dists = l2_dist.unsqueeze(-1)
                else:
                    l2_dists = torch.cat((l2_dists, l2_dist.unsqueeze(-1)))
            return torch.mean(l2_dists)
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]
            num_classes = outputs.size()[1]
            outputs = F.softmax(outputs, dim=1)
            targets = torch.eye(num_classes)
            l2_dists = None
            for output, label in zip(outputs, labels):
                target = targets[int(label.item())]
                l2_dist = torch.dist(target, output.cpu())
                if l2_dists is None:
                    l2_dists = l2_dist.unsqueeze(-1)
                else:
                    l2_dists = torch.cat((l2_dists, l2_dist.unsqueeze(-1)))
            return l2_dists
    return fn

import torch
import torch.nn as nn


def log_sum_exp(x):
    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = log_sum_exp(logits) - class_select(logits, target)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        return cross_entropy_with_weights(input, target, weights)
