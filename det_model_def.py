from determined.pytorch import DataLoader, PyTorchTrial
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *

class CIFAR10Trial(PyTorchTrial):
    def __init__(self, context: det.TrialContext):
        # Initialize the trial class.
        pass

    def build_model(self):
        # Build the model.
        net = ResNet18()
        return net

    def optimizer(self, model: nn.Module):
        # Define the optimizer.
        pass

    def train_batch(self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int):
        # Define the training forward pass and calculate loss and other metrics
        # for a batch of training data.
        pass

    def evaluate_batch(self, batch: TorchData, model: nn.Module):
        # Define how to evaluate the model by calculating loss and other metrics.
        # for a batch of validation data.
        pass

    def build_training_data_loader(self):
        # Create the training data loader.
        # This should return a determined.pytorch.Dataset.
        pass

    def build_validation_data_loader(self):
        # Create the validation data loader.
        # This should return a determined.pytorch.Dataset.
        pass
