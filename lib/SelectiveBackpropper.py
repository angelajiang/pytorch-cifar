import backproppers
import loggers
import selectors
import torch
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
        sb_start_epoch = 1
        log_interval = 1
        sampling_max = 1
        # Params for resuming from checkpoint
        start_epoch = 0
        start_num_backpropped = 0
        start_num_skipped = 0

        probability_calculator = selectors.SelectProbabiltyCalculator(sampling_min,
                                                                      sampling_max,
                                                                      num_classes,
                                                                      device,
                                                                      square=False,
                                                                      translate=False)

        final_selector = selectors.DeterministicSamplingSelector(probability_calculator,
                                                                 initial_sum=1)
        final_backpropper = backproppers.SamplingBackpropper(device,
                                                             model,
                                                             optimizer)
        self.selector = selectors.PrimedSelector(selectors.BaselineSelector(),
                                                 final_selector,
                                                 sb_start_epoch,
                                                 epoch=start_epoch)
        self.backpropper = backproppers.PrimedBackpropper(backproppers.BaselineBackpropper(device,
                                                                                           model,
                                                                                           optimizer),
                                                     final_backpropper,
                                                     sb_start_epoch,
                                                     epoch=start_epoch)
        self.trainer = trainer.Trainer(device,
                                       model,
                                       self.selector,
                                       self.backpropper,
                                       batch_size,
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
        self.selector.next_epoch()
        self.backpropper.next_epoch()

    def next_partition(self):
        self.logger.next_partition()
