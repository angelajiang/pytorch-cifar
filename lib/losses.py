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

            max_outputs = None
            for output, label in zip(outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_output = torch.max(other_outputs)
                if max_outputs is None:
                    max_outputs = max_output.unsqueeze(-1)
                else:
                    max_outputs = torch.cat((max_outputs, max_output.unsqueeze(-1)))
            cross_entropy_loss = -torch.mean(class_outputs + max_outputs) / 200.
            return cross_entropy_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            max_outputs = None
            for output, label in zip(outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_output = torch.max(other_outputs)
                if max_outputs is None:
                    max_outputs = max_output.unsqueeze(-1)
                else:
                    max_outputs = torch.cat((max_outputs, max_output.unsqueeze(-1)))
            cross_entropy_loss = -(class_outputs + max_outputs) / 200.
            return cross_entropy_loss
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
