import numpy as np

# TODO: Transform into base classes
def get_selector(selector_type, num_images_to_prime):
    if selector_type == "threshold":
        final_selector = ThresholdSelector()
    elif selector_type == "alwayson":
        final_selector = AlwaysOnSelector()
    elif selector_type == "stale":
        final_selector = StaleSelector()
    else:
        print("FP Selector must be in {alwayson, threshold}")
        exit()
    selector = PrimedSelector(AlwaysOnSelector(),
                              final_selector,
                              num_images_to_prime)
    return selector

class PrimedSelector(object):
    def __init__(self, initial, final, initial_num_images, epoch=0):
        self.initial = initial
        self.final = final
        self.initial_num_images = initial_num_images
        self.num_trained = 0

    def next_partition(self, partition_size):
        self.num_trained += partition_size

    def get_selector(self):
        return self.initial if self.num_trained < self.initial_num_images else self.final

    def select(self, *args, **kwargs):
        return self.get_selector().select(*args, **kwargs)

    def mark(self, *args, **kwargs):
        return self.get_selector().mark(*args, **kwargs)


class AlwaysOnSelector():
    def select(self, probability):
        draw = np.random.uniform(0, 1)
        return draw < probability

    def mark(self, examples):
        for example in examples:
            example.forward_select_probability = 1.
            example.forward_select = self.select(example.forward_select_probability)
        return examples

class StaleSelector():
    def __init__(self):
        self.threshold = 2
        self.logger = {"counter": 0, "forward": 0, "no_forward": 0}
        print("StaleSelector_{}".format(self.threshold))

    def select(self, example):
        if self.logger['counter'] % 10000 == 0:
            print(self.logger)
        self.logger['counter'] += 1

        example.epochs_since_update += 1
        if example.epochs_since_update >= self.threshold:
            self.logger['forward'] += 1
            return True
        else:
            self.logger['no_forward'] += 1
            return False

    def mark(self, examples):
        for example in examples:
            example.forward_select = self.select(example)
        return examples

class ThresholdSelector():
    def __init__(self):
        self.logger = {"counter": 0, "path_3": 0, "path_2": 0, "path_1": 0}
        self.historical_sps = {}
        self.times_passed = {}
        self.threshold = 0.0001
        self.times_passed_threshold = 5
        print("ThesholdSelector {}-{}".format(self.threshold, self.times_passed_threshold))

    def select(self, example):

        if self.logger['counter'] % 10000 == 0:
            print(self.logger)
        self.logger['counter'] += 1

        image_id = example.image_id

        # First time seeing image. No SP calculated yet. FP image.
        if image_id not in self.times_passed.keys():
            self.historical_sps[image_id] = None
            self.times_passed[image_id] = 0
            return True

        times_passed = self.times_passed[image_id]
        # Image was forward propped last time. Update history with SP.
        if times_passed == 0:
            self.historical_sps[image_id] = example.select_probability
            self.logger['path_1'] += 1

        last_sp = self.historical_sps[image_id]
        if last_sp < self.threshold and times_passed <= self.times_passed_threshold:
            self.times_passed[image_id] += 1
            self.logger['path_2'] += 1
            return False
        else:
            self.times_passed[image_id] = 0
            self.logger['path_3'] += 1
            return True

    def mark(self, examples):
        for example in examples:
            example.forward_select = self.select(example)
        return examples
