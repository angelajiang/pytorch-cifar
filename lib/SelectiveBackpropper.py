import backproppers
import loggers
import selectors
import torch
import torch.nn as nn
import trainer as trainer

class SelectiveBackpropper:

    def __init__(self,
                 model,
                 optimizer,
                 sampling_min,
                 batch_size,
                 lr_sched,
                 num_classes,
                 forwardlr):

        ## Hardcoded params
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        images_to_prime = 50000
        log_interval = 1
        sampling_max = 1
        max_history_len = 1024
        prob_loss_fn = nn.CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss
        # Params for resuming from checkpoint
        start_epoch = 0
        start_num_backpropped = 0
        start_num_skipped = 0

        probability_calculator = selectors.RelativeCubedProbabilityCalculator(device,
                                                                             prob_loss_fn,
                                                                             sampling_min,
                                                                             max_history_len)
        final_selector = selectors.SamplingSelector(probability_calculator)
        final_backpropper = backproppers.SamplingBackpropper(device,
                                                                 model,
                                                                 optimizer,
                                                                 loss_fn)

        self.selector = selectors.PrimedSelector(selectors.BaselineSelector(),
                                                 final_selector,
                                                 images_to_prime)
        self.backpropper = backproppers.PrimedBackpropper(backproppers.BaselineBackpropper(device,
                                                                                           model,
                                                                                           optimizer,
                                                                                           loss_fn),
                                                     final_backpropper,
                                                     images_to_prime)
        self.trainer = trainer.Trainer(device,
                                       model,
                                       self.selector,
                                       self.backpropper,
                                       batch_size,
                                       loss_fn,
                                       lr_schedule=lr_sched,
                                       forwardlr=forwardlr)

        self.logger = loggers.Logger(log_interval = log_interval,
                                     epoch=start_epoch,
                                     num_backpropped=start_num_backpropped,
                                     num_skipped=start_num_skipped)
        self.trainer.on_forward_pass(self.logger.handle_forward_batch)
        self.trainer.on_backward_pass(self.logger.handle_backward_batch)

    def next_epoch(self):
        self.logger.next_epoch()

    def next_partition(self):
        self.logger.next_partition()
