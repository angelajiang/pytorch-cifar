import sys
import ctypes
import torch.multiprocessing as mp
import json
import numpy as np
import threading
import torch
import torch.nn as nn
import lib.forwardproppers as forwardproppers
import lib.sb_util as sb_util
import lib.ringbuffer as ringbuffer
import time
from copy import deepcopy
from torch.multiprocessing import Process, Queue

def writer(image_ids, forwardpropper):
    print("Writer starting")
    for i in range(0, 1000):
      image_ids[i] = 1
    print("Writer done")


def selector_process(image_ids_tensor, forwardpropper, mark_batch_fn, selector_net, trainloader):
    print("[selector] started")

    forwardpropper.net = selector_net

    image_id_tensor_index = 0
    for i, batch in enumerate(trainloader):
        if i == len(trainloader) - 1:
            final = True
        else:
            final = False
        annotated_forward_batch = mark_batch_fn(batch, final=final)
        if annotated_forward_batch is not None:
            image_ids = [em.example.image_id for em in annotated_forward_batch]
            for image_id in image_ids:
                image_ids_tensor[image_id_tensor_index] = image_id
                image_id_tensor_index += 1
    print("[selector] writer done")

class ExampleAndMetadata(object):
    def __init__(self, example, metadata):
        self.example = example
        self.metadata = metadata

class Example(object):
    # TODO: Add ExampleCollection class
    def __init__(self,
                 loss=None,
                 output=None,
                 softmax_output=None,
                 target=None,
                 datum=None,
                 image_id=None,
                 select_probability=None):
        if loss is not None:
            self.loss = loss.detach().cpu()
        if output is not None:
            self.output = output.detach().cpu()
        if softmax_output is not None:
            self.softmax_output = softmax_output.detach().cpu()
        self.forward_select = True
        self.target = target.detach().cpu()
        self.datum = datum.detach().cpu()
        self.image_id = image_id
        self.select_probability = select_probability
        self.backpropped_loss = None   # Populated after backprop

    def __str__(self):
        string = "Image {}\n\ndatum:{}\ntarget:{}\nsp:{}\n".format(self.image_id,
                                                                   self.datum,
                                                                   self.target,
                                                                   self.select_probability)
        if hasattr(self, 'loss'):
            string += "loss:{}\n".format(self.loss)
        if hasattr(self, 'output'):
            string += "output:{}\n".format(self.output)
        if hasattr(self, 'softmax_output'):
            string += "softmax_output:{}\n".format(self.softmax_output)

        return string


    @property
    def predicted(self):
        _, predicted = self.softmax_output.max(0)
        return predicted

    @property
    def is_correct(self):
        return self.predicted.eq(self.target)


class Trainer(object):
    def __init__(self,
                 device,
                 net,
                 selector,
                 backpropper,
                 batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):
        self.device = device
        self.net = net
        self.selector = selector
        self.backpropper = backpropper
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.backprop_queue = []
        self.forward_pass_handlers = []
        self.backward_pass_handlers = []
        self.global_num_backpropped = 0
        self.global_num_forwards = 0
        self.global_num_analyzed = 0
        self.forwardlr = forwardlr
        self.max_num_backprops = max_num_backprops
        self.on_backward_pass(self.update_num_backpropped)
        self.on_forward_pass(self.update_num_forwards)
        self.on_forward_pass(self.update_num_analyzed)
        self.example_metadata = {}
        if lr_schedule:
            self.load_lr_schedule(lr_schedule)
            self.on_backward_pass(self.update_learning_rate)

    def update_num_backpropped(self, batch):
        self.global_num_backpropped += sum([1 for em in batch if em.example.select])

    def update_num_forwards(self, batch):
        self.global_num_forwards += sum([1 for em in batch if em.example.forward_select])

    def update_num_analyzed(self, batch):
        self.global_num_analyzed += len(batch)

    def on_forward_pass(self, handler):
        self.forward_pass_handlers.append(handler)

    def on_backward_pass(self, handler):
        self.backward_pass_handlers.append(handler)

    def emit_forward_pass(self, batch):
        for handler in self.forward_pass_handlers:
            handler(batch)

    def emit_backward_pass(self, batch):
        for handler in self.backward_pass_handlers:
            handler(batch)

    # TODO move to a LRScheduler object or to backpropper
    def load_lr_schedule(self, schedule_path):
        with open(schedule_path, "r") as f:
            data = json.load(f)
        self.lr_schedule = {}
        for k in data:
            self.lr_schedule[int(k)] = data[k]

    def set_learning_rate(self, lr):
        print("Setting learning rate to {} at {} backprops".format(lr,
                                                                   self.global_num_backpropped))
        for param_group in self.backpropper.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def counter(self):
        if self.forwardlr:
            counter = self.global_num_analyzed
        else:
            counter = self.global_num_backpropped
        return counter

    def update_learning_rate(self, batch):
        for start_num_backprop in reversed(sorted(self.lr_schedule)):
            lr = self.lr_schedule[start_num_backprop]
            if self.counter >= start_num_backprop:
                if self.backpropper.optimizer.param_groups[0]['lr'] is not lr:
                    self.set_learning_rate(lr)
                break

    @property
    def stopped(self):
        return self.counter >= self.max_num_backprops

    def train(self, trainloader):
        for i, batch in enumerate(trainloader):
            if self.stopped: break
            if i == len(trainloader) - 1:
                self.train_batch(batch, final=True)
            else:
                self.train_batch(batch, final=False)

    def train_batch(self, batch, final):
        pass

    def forward_pass(self, data, targets, image_ids):
        pass

    def get_batch(self, final):
        num_images_to_backprop = 0
        for index, em in enumerate(self.backprop_queue):
            num_images_to_backprop += int(em.example.select)
            if num_images_to_backprop == self.batch_size:
                # Note: includes item that should and shouldn't be backpropped
                backprop_batch = self.backprop_queue[:index+1]
                self.backprop_queue = self.backprop_queue[index+1:]
                return backprop_batch
        if final:
            def get_num_to_backprop(batch):
                return sum([1 for em in batch if em.example.select])
            backprop_batch = self.backprop_queue
            self.backprop_queue = []
            if get_num_to_backprop(backprop_batch) == 0:
                return None
            return backprop_batch
        return None

    def create_example_batch(self, data, targets, image_ids):
        batch = []
        for target, datum, image_id in zip(targets, data, image_ids):
            image_id = image_id.item()
            if image_id not in self.example_metadata:
                self.example_metadata[image_id] = {"epochs_since_update": 0}
            example = Example(target=target, datum=datum, image_id=image_id, select_probability=1)
            example.select = True
            batch.append(ExampleAndMetadata(example, self.example_metadata[image_id]))
        return batch

class MemoizedTrainer(Trainer):
    def __init__(self,
                 device,
                 net,
                 selector,
                 fp_selector,
                 backpropper,
                 batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):

        super(MemoizedTrainer, self).__init__(device,
                                net,
                                selector,
                                backpropper,
                                batch_size,
                                loss_fn,
                                max_num_backprops,
                                lr_schedule,
                                forwardlr)

        self.fp_selector = fp_selector
        self.forward_queue = []
        self.forward_batch_size = batch_size
        self.forwardpropper = forwardproppers.CutoutForwardpropper(device,
                                                                   net,
                                                                   loss_fn)

    def train_batch(self, batch, final):
        EMs = self.create_example_batch(*batch)
        batch_marked_for_fp = self.fp_selector.mark(EMs)
        self.forward_queue += batch_marked_for_fp
        batch_to_fp = self.get_forward_batch(final)
        if batch_to_fp:
            forward_pass_batch = self.forwardpropper.forward_pass(batch_to_fp)
            annotated_forward_batch = self.selector.mark(forward_pass_batch)
            self.emit_forward_pass(annotated_forward_batch)
            self.backprop_queue += annotated_forward_batch
            backprop_batch = self.get_batch(final)
            if backprop_batch:
                annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
                self.emit_backward_pass(annotated_backward_batch)

    def get_forward_batch(self, final):
        num_images_to_fp = 0
        max_queue_size = self.forward_batch_size * 4
        for index, em in enumerate(self.forward_queue):
            num_images_to_fp += int(em.example.forward_select)
            if num_images_to_fp == self.forward_batch_size:
                # Note: includes item that should and shouldn't be forward propped
                forward_batch = self.forward_queue[:index+1]
                self.forward_queue = self.forward_queue[index+1:]
                return forward_batch
        if final or len(self.forward_queue) > max_queue_size:
            forward_batch = self.forward_queue
            self.forward_queue = []
            return forward_batch
        return None

class QueueElement:
    def __init__(self, examples_and_metadata, final):
        self.em = examples_and_metadata
        self.final = final

class AsyncTrainer(MemoizedTrainer):
    def __init__(self,
                 device1,
                 device2,
                 net,
                 selector,
                 fp_selector,
                 backpropper,
                 batch_size,
                 loss_fn,
                 num_images,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):

        self.device1 = device1
        self.device2 = device2
        assert self.device1 != self.device2

        self.net = net
        self.selector = selector
        self.backpropper = backpropper
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.backprop_queue = []
        self.forward_pass_handlers = []
        self.backward_pass_handlers = []
        self.global_num_backpropped = 0
        self.global_num_forwards = 0
        self.global_num_analyzed = 0
        self.forwardlr = forwardlr
        self.max_num_backprops = max_num_backprops
        self.on_backward_pass(self.update_num_backpropped)
        self.on_forward_pass(self.update_num_forwards)
        self.on_forward_pass(self.update_num_analyzed)
        self.example_metadata = {}
        if lr_schedule:
            self.load_lr_schedule(lr_schedule)
            self.on_backward_pass(self.update_learning_rate)

        self.fp_selector = fp_selector
        self.forward_queue = []
        self.forward_batch_size = batch_size
        self.forwardpropper = forwardproppers.CutoutForwardpropper(self.device2,
                                                                   net,
                                                                   loss_fn)
        self.first = True
        self.examples = {}

    def prep_async(self, trainloader):
        for i, batch in enumerate(trainloader):
            EMs = self.create_example_batch(*batch)
            for em in EMs:
                image_id = em.example.image_id
                self.examples[image_id] = em

    def train(self, trainloader):

        print("[train] Started train for new epoch")
        selector_net = deepcopy(self.net)
        print("[train] finished copy")
        selector_net.to(self.device2)
        print("[train] finished network move")

        image_ids_tensor = torch.zeros(50000)
        image_ids_tensor.share_memory_()
        image_ids_tensor.fill_(-1)

        writer_process = mp.Process(target=selector_process, args=(image_ids_tensor, self.forwardpropper, self.mark_batch, selector_net, trainloader, ))
        writer_process.start()

        if self.first:
            self.prep_async(trainloader)
            self.first = False

        i = 0
        while True:
            if image_ids_tensor[i] == -1:
                a = 1
            else:
                self.backprop_queue.append(self.examples[image_ids_tensor[i].item()])
                backprop_batch = self.get_batch(False)
                if backprop_batch:
                    print("[train] starting a backprop")
                    annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
                    self.emit_backward_pass(annotated_backward_batch)
                i += 1

    def mark_batch(self, batch, final):
        EMs = self.create_example_batch(*batch)
        batch_marked_for_fp = self.fp_selector.mark(EMs)
        self.forward_queue += batch_marked_for_fp
        batch_to_fp = self.get_forward_batch(final)
        if batch_to_fp:
            print("[mark_batch] doing a forward pass")
            forward_pass_batch = self.forwardpropper.forward_pass(batch_to_fp)
            annotated_forward_batch = self.selector.mark(forward_pass_batch)
            self.emit_forward_pass(annotated_forward_batch)
            return annotated_forward_batch
        return None

    def get_forward_batch(self, final):
        num_images_to_fp = 0
        max_queue_size = self.forward_batch_size * 4
        for index, em in enumerate(self.forward_queue):
            num_images_to_fp += int(em.example.forward_select)
            if num_images_to_fp == self.forward_batch_size:
                # Note: includes item that should and shouldn't be forward propped
                forward_batch = self.forward_queue[:index+1]
                self.forward_queue = self.forward_queue[index+1:]
                return forward_batch
        if final or len(self.forward_queue) > max_queue_size:
            forward_batch = self.forward_queue
            self.forward_queue = []
            return forward_batch
        return None


class NoFilterTrainer(Trainer):
    def __init__(self,
                 device,
                 net,
                 backpropper,
                 batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):

        super(NoFilterTrainer, self).__init__(device,
                                net,
                                None,
                                backpropper,
                                batch_size,
                                loss_fn,
                                max_num_backprops,
                                lr_schedule,
                                forwardlr)

    def train_batch(self, batch, final):
        annotated_forward_batch = self.create_example_batch(*batch)
        self.backprop_queue += annotated_forward_batch
        backprop_batch = self.get_batch(final)
        if backprop_batch:
            annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
            self.emit_backward_pass(annotated_backward_batch)
            self.emit_forward_pass(annotated_backward_batch)

class KathTrainer(Trainer):
    def __init__(self,
                 device,
                 net,
                 backpropper,
                 batch_size,
                 pool_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):
        super(KathTrainer, self).__init__(device,
                                          net,
                                          None,
                                          backpropper,
                                          batch_size,
                                          loss_fn,
                                          max_num_backprops,
                                          lr_schedule,
                                          forwardlr)
        self.condition = kath_util.VarianceReductionCondition()
        self.pool = []
        self.pool_size = pool_size

    def train(self, trainloader):
        if self.condition.satified:
            for i, batch in enumerate(trainloader):
                forward_pass_batch = self.forward_pass(*batch)
                self.emit_forward_pass(forward_pass_batch)
                self.pool += forward_pass_batch
                if len(self.pool) >= self.pool_size:
                    self.train_pool(self.pool)
                    self.pool = []
        else:
            for i, batch in enumerate(trainloader):
                forward_pass_batch = self.forward_pass(*batch)
                self.emit_forward_pass(forward_pass_batch)
                self.pool += forward_pass_batch
                if len(self.pool) >= self.batch_size:
                    self.train_all(self.pool)
                    self.pool = []

    def train_pool(self, pool):
        backprop_batch = self.get_batch(pool)
        annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
        self.emit_backward_pass(annotated_backward_batch)

    def train_all(self, pool):
        for em in pool:
            em.example.select = True
        annotated_backward_batch = self.backpropper.backward_pass(pool)
        self.emit_backward_pass(annotated_backward_batch)

    def get_probabilities(self, pool):
        loss_sum = sum([example.loss.item() for example in pool])
        probs = [example.loss.item() / loss_sum for example in pool]
        return probs

    def get_batch(self, examples_and_metadata):
        pool = [em.example for em in examples_and_metadata]
        probs = self.get_probabilities(pool)
        for example, prob in zip(pool, probs):
            example.select_probability = prob
            example.select = False

        # Sample batch_size with replacement
        chosen_examples = np.random.choice(pool, self.batch_size, replace=True, p=probs)

        # Populate batch with sampled_choices
        for example in chosen_examples:
            example.select = True

        return examples_and_metadata

    def forward_pass(self, data, targets, image_ids):
        data, targets = data.to(self.device), targets.to(self.device)

        self.net.eval()
        with torch.no_grad():
            outputs = self.net(data)

        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)

        examples = zip(losses, outputs, softmax_outputs, targets, data, image_ids)
        return [ExampleAndMetadata(Example(*example), {}) for example in examples]


class KathBaselineTrainer(KathTrainer):
    def __init__(self,
                 device,
                 net,
                 backpropper,
                 batch_size,
                 pool_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):

        super(KathBaselineTrainer, self).__init__(device,
                                                  net,
                                                  backpropper,
                                                  batch_size,
                                                  pool_size,
                                                  loss_fn,
                                                  max_num_backprops=float('inf'),
                                                  lr_schedule=None,
                                                  forwardlr=False)

    def get_probabilities(self, pool):
        loss_sum = sum([example.loss.item() for example in pool])
        probs = [1. / len(pool) for example in pool]
        return probs


