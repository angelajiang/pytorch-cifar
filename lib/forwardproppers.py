
import torch
import torch.nn as nn

class BaselineForwardpropper(object):

    def __init__(self, device, net, dataset, optimizer, loss_fn):
        self.optimizer = optimizer
        self.net = net
        self.dataset = dataset
        self.device = device
        self.loss_fn = loss_fn
        # TODO: This doesn't work after resuming from checkpoint

    def _get_chosen_image_ids(self, batch):
        return [example.image_id for example in batch if example.get_select(True)]

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [example.datum for example in batch if example.get_select(True)]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [example.target for example in batch if example.get_select(True)]
        chosen = len([example.target for example in batch if example.get_select(True)])
        whole = len(batch)
        return torch.stack(chosen_targets)

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

    def forward_pass(self, batch):

        image_ids = self._get_chosen_image_ids(batch)
        data = self._get_chosen_data_tensor(batch)
        targets = self._get_chosen_targets_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        #self.net.eval()
        #with torch.no_grad():
        #    outputs = self.net(data)
        outputs = self.net(data)

        losses = self.loss_fn(reduce=False)(outputs, targets)
        softmax_outputs = nn.Softmax()(outputs)

        _, predicted = outputs.max(1)
        is_corrects = predicted.eq(targets)

        examples = []
        for image_id, loss, output, softmax_output, is_correct in zip(image_ids,
                                                                      losses,
                                                                      outputs,
                                                                      softmax_outputs,
                                                                      is_corrects):
            e = self.dataset.examples[image_id]
            e.loss = loss
            e.output = output
            e.softmax_output = softmax_output
            e.correct = is_correct.item()
            examples.append(e)

        return examples
