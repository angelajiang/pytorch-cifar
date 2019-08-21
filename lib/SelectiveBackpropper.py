import backproppers
import calculators
import loggers
import selectors
import time
import torch
import torch.nn as nn
import trainer as trainer

start_time_seconds = time.time()

class SelectiveBackpropper:

    def __init__(self,
                 model,
                 optimizer,
                 sampling_min,
                 batch_size,
                 lr_sched,
                 num_classes,
                 forwardlr,
                 kath):

        ## Hardcoded params
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_images_to_prime = 50000
        log_interval = 1
        sampling_max = 1
        max_history_len = 1024
        prob_loss_fn = nn.CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss
        prob_pow = 3
        sample_size = 0 # only needed for kath, topk, lowk
        # Params for resuming from checkpoint
        start_epoch = 0
        start_num_backpropped = 0
        start_num_skipped = 0
        kath_oversampling_rate = 4

        if kath:
            self.selector = None
            final_backpropper = backproppers.BaselineBackpropper(device,
                                                                     model,
                                                                     optimizer,
                                                                     loss_fn)
            self.backpropper = backproppers.PrimedBackpropper(backproppers.BaselineBackpropper(device,
                                                                                                  model,
                                                                                                  optimizer,
                                                                                                  loss_fn),
                                                             final_backpropper,
                                                             num_images_to_prime)
            self.trainer = trainer.KathTrainer(device,
                                               model,
                                               self.backpropper,
                                               batch_size,
                                               batch_size * kath_oversampling_rate,
                                               loss_fn,
                                               lr_sched,
                                               forwardlr=forwardlr)
        else:

            probability_calculator = calculators.get_probability_calculator("relative",
                                                                            device,
                                                                            prob_loss_fn,
                                                                            sampling_min,
                                                                            sampling_max,
                                                                            num_classes,
                                                                            max_history_len,
                                                                            prob_pow)
            self.selector = selectors.get_selector("sampling",
                                                   probability_calculator,
                                                   num_images_to_prime,
                                                   sample_size)

            final_backpropper = backproppers.SamplingBackpropper(device,
                                                                 model,
                                                                 optimizer,
                                                                 loss_fn)

            self.backpropper = backproppers.PrimedBackpropper(backproppers.BaselineBackpropper(device,
                                                                                               model,
                                                                                               optimizer,
                                                                                               loss_fn),
                                                              final_backpropper,
                                                              num_images_to_prime)
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
                                     num_skipped=start_num_skipped,
                                     start_time_seconds = start_time_seconds)

        self.trainer.on_forward_pass(self.logger.handle_forward_batch)
        self.trainer.on_backward_pass(self.logger.handle_backward_batch)

        #self.backpropper.next_partition(50000)
        #if self.selector:
        #    self.selector.next_partition(50000)

    def next_epoch(self):
        self.logger.next_epoch()

    def next_partition(self):
        self.logger.next_partition()
