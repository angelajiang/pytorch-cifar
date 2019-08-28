import json
import numpy as np
import torch
import torch.nn as nn
import forwardproppers
import sb_util


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
        self.target = target.detach()
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
        self.forwardlr = forwardlr
        self.max_num_backprops = max_num_backprops
        self.on_backward_pass(self.update_num_backpropped)
        self.on_forward_pass(self.update_num_forwards)
        if lr_schedule:
            self.load_lr_schedule(lr_schedule)
            self.on_backward_pass(self.update_learning_rate)

    def update_num_backpropped(self, batch):
        self.global_num_backpropped += sum([1 for e in batch if e.select])

    def update_num_forwards(self, batch):
        self.global_num_forwards += len(batch)

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
            counter = self.global_num_forwards
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
        forward_pass_batch = self.forward_pass(*batch)
        annotated_forward_batch = self.selector.mark(forward_pass_batch)
        self.emit_forward_pass(annotated_forward_batch)
        self.backprop_queue += annotated_forward_batch
        backprop_batch = self.get_batch(final)
        if backprop_batch:
            annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
            self.emit_backward_pass(annotated_backward_batch)

    def forward_pass(self, data, targets, image_ids):
        data, targets = data.to(self.device), targets.to(self.device)

        self.net.eval()
        with torch.no_grad():
            outputs = self.net(data)

        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)

        examples = zip(losses, outputs, softmax_outputs, targets, data, image_ids)
        return [Example(*example) for example in examples]

    def get_batch(self, final):

        num_images_to_backprop = 0
        for index, example in enumerate(self.backprop_queue):
            num_images_to_backprop += int(example.select)
            if num_images_to_backprop == self.batch_size:
                # Note: includes item that should and shouldn't be backpropped
                backprop_batch = self.backprop_queue[:index+1]
                self.backprop_queue = self.backprop_queue[index+1:]
                return backprop_batch
        if final:
            def get_num_to_backprop(batch):
                return sum([1 for example in batch if example.select])
            backprop_batch = self.backprop_queue
            self.backprop_queue = []
            if get_num_to_backprop(backprop_batch) == 0:
                return None
            return backprop_batch
        return None

class StaleTrainer(Trainer):
    def __init__(self,
                 device,
                 net,
                 selector,
                 backpropper,
                 batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False,
                 fp_selector_type="alwayson"):

        super(StaleTrainer, self).__init__(device,
                                           net,
                                           selector,
                                           backpropper,
                                           batch_size,
                                           loss_fn,
                                           max_num_backprops,
                                           lr_schedule,
                                           forwardlr)

        self.forwardpropper = forwardproppers.CutoutForwardpropper(device,
                                                                   net,
                                                                   loss_fn)
        self.examples = {}

    def create_example_batch(self, data, targets, image_ids):
        data, targets = data.to(self.device), targets.to(self.device)
        batch = []
        for target, datum, image_id in zip(targets, data, image_ids):
            image_id = image_id.item()
            if image_id not in self.examples.keys():
                example = Example(target=target, datum=datum, image_id=image_id, select_probability=1)
                example.select = True
                self.examples[image_id] = example
            else:
                example = self.examples[image_id]
                example.datum = datum.detach()
            batch.append(example)
            
        return batch

    def train_batch(self, batch, final):
        examples = self.create_example_batch(*batch)
        forward_pass_batch = self.forwardpropper.forward_pass(examples)
        annotated_forward_batch = self.selector.mark(forward_pass_batch)
        self.emit_forward_pass(annotated_forward_batch)
        self.backprop_queue += annotated_forward_batch
        backprop_batch = self.get_batch(final)
        if backprop_batch:
            annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
            self.emit_backward_pass(annotated_backward_batch)

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

        self.on_backward_pass(self.update_num_forwards)

    def train_batch(self, batch, final):
        annotated_forward_batch = self.create_example_batch(*batch)
        self.backprop_queue += annotated_forward_batch
        backprop_batch = self.get_batch(final)
        if backprop_batch:
            annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
            self.emit_backward_pass(annotated_backward_batch)
            self.emit_forward_pass(annotated_backward_batch)

    def create_example_batch(self, data, targets, image_ids):
        # data, targets = data.to(self.device), targets.to(self.device)
        batch = []
        for target, datum, image_id in zip(targets, data, image_ids):
            example = Example(target=target, datum=datum, image_id=image_id, select_probability=1)
            example.select = True
            batch.append(example)
        return batch

class MemoizedTrainer(Trainer):
    def __init__(self,
                 device,
                 net,
                 dataset,
                 bp_selector,
                 backpropper,
                 bp_batch_size,
                 fp_selector,
                 forwardpropper,
                 forward_batch_size,
                 loss_fn,
                 max_num_backprops=float('inf'),
                 lr_schedule=None,
                 forwardlr=False):

        super(MemoizedTrainer, self).__init__(device,
                                net,
                                dataset,
                                bp_selector,
                                backpropper,
                                bp_batch_size,
                                loss_fn,
                                max_num_backprops,
                                lr_schedule,
                                forwardlr)

        self.forward_queue = []
        self.forwardpropper = forwardpropper
        self.forward_batch_size = forward_batch_size
        self.fp_selector = fp_selector
        self.forward_mark_handlers = []

    def on_forward_mark(self, handler):
        self.forward_mark_handlers.append(handler)

    def emit_forward_mark(self, batch):
        for handler in self.forward_mark_handlers:
            handler(batch)

    def count_allocated_tensors(self):
        import gc
        count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    #print(type(obj), obj.size())
                    count += 1
            except:
                pass 
        return count

    def train_batch(self, candidate_forward_batch, final):
        # Transform candidate forward_batch into examples
        candidate_forward_batch_examples = []
        for datum, image_id in zip(candidate_forward_batch[0], candidate_forward_batch[2]):
            e = self.dataset.examples[image_id.item()]
            e.datum = datum             # OPT: image copy?
            candidate_forward_batch_examples.append(e)

        batch_marked_for_fp = self.fp_selector.mark(candidate_forward_batch_examples) # OPT: in place?
        self.emit_forward_mark(batch_marked_for_fp)
        self.forward_queue += batch_marked_for_fp
        batch_to_fp = self.get_forward_batch(final)
        if batch_to_fp:
            candidate_backward_batch = self.forward_pass(batch_to_fp)
            self.emit_forward_pass(candidate_backward_batch)

            batch_marked_for_bp = self.selector.mark(candidate_backward_batch) # OPT: in place?
            #print([a.get_select(False) for a in batch_marked_for_bp])
            self.backprop_queue += batch_marked_for_bp
            batch_to_bp = self.get_batch(final)
            if batch_to_bp:
                annotated_backward_batch = self.backpropper.backward_pass(batch_to_bp)
                self.emit_backward_pass(annotated_backward_batch)

    def get_forward_batch(self, final):
        num_images_to_fp = 0
        for index, example in enumerate(self.forward_queue):
            num_images_to_fp += int(example.get_select(True))
            if num_images_to_fp == self.forward_batch_size:
                # Note: includes item that should and shouldn't be forward propped
                forward_batch = self.forward_queue[:index+1]
                self.forward_queue = self.forward_queue[index+1:]
                return forward_batch
        if final:
            def get_num_to_forward(batch):
                return sum([1 for example in batch if example.get_select(True)])
            forward_batch = self.forward_queue
            self.forward_queue = []
            if get_num_to_forward(forward_batch) == 0:
                return None
            return forward_batch
        return None


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
        self.pool = []
        self.pool_size = pool_size

    def train(self, trainloader):
        for i, batch in enumerate(trainloader):
            forward_pass_batch = self.forward_pass(*batch)
            self.emit_forward_pass(forward_pass_batch)
            self.pool += forward_pass_batch
            if len(self.pool) >= self.pool_size:
                self.train_pool(self.pool)
                self.pool = []

    def train_pool(self, pool):
        backprop_batch = self.get_batch(pool)
        annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
        self.emit_backward_pass(annotated_backward_batch)

    def get_probabilities(self, pool):
        loss_sum = sum([example.loss.item() for example in pool])
        probs = [example.loss.item() / loss_sum for example in pool]
        return probs

    def get_batch(self, pool):
        probs = self.get_probabilities(pool)
        for example, prob in zip(pool, probs):
            example.select_probability = prob

        # Sample batch_size with replacement
        chosen_examples = np.random.choice(pool, self.batch_size, replace=False, p=probs)

        # Populate batch with sampled_choices
        for example in chosen_examples:
            example.select = True

        return chosen_examples

    def forward_pass(self, data, targets, image_ids):
        data, targets = data.to(self.device), targets.to(self.device)

        self.net.eval()
        with torch.no_grad():
            outputs = self.net(data)

        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)

        examples = zip(losses, outputs, softmax_outputs, targets, data, image_ids)
        return [Example(*example) for example in examples]

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


