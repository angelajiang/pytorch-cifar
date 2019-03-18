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
