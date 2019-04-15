import numpy as np
import torch.nn.functional as F
import torch


def CrossEntropyReweightedLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.softmax(outputs, dim=1)       # compute the softmax values
            num_classes = outputs.size()[1]            # num classes
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            losses = None
            for softmax_output, class_prob, label in zip(outputs, class_outputs, labels):

                # Hot encode
                target_vector = np.zeros(num_classes)
                target_vector[label.item()] = 1
                target_tensor = torch.Tensor(target_vector)

                # Calculate weight based on L2 fully proportional
                l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
                weight = l2_dist / 2.
                print(weight, -torch.log(class_prob))

                # Calculate loss
                loss = -torch.log(class_prob) * weight
                if losses is None:
                    losses = loss.unsqueeze(-1)
                else:
                    losses = torch.cat((losses, loss.unsqueeze(-1)))
            reduced_loss = torch.mean(losses)
            return reduced_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.softmax(outputs, dim=1)       # compute the softmax values
            num_classes = outputs.size()[1]            # num classes
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            losses = None
            for softmax_output, class_prob, label in zip(outputs, class_outputs, labels):
                # Hot encode
                target_vector = np.zeros(num_classes)
                target_vector[label.item()] = 1
                target_tensor = torch.Tensor(target_vector)

                # Calculate weight based on L2 fully proportional
                l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
                weight = l2_dist / 2.
                print(weight, -torch.log(class_prob))

                # Calculate loss
                loss = -torch.log(class_prob) * weight
                if losses is None:
                    losses = loss.unsqueeze(-1)
                else:
                    losses = torch.cat((losses, loss.unsqueeze(-1)))
            return losses
    return fn

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
