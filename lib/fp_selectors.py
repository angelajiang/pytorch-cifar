import numpy as np

# TODO: Transform into base classes
def get_selector(selector_type):
    if selector_type == "threshold":
        final_selector = ThresholdSelector()
    elif selector_type == "alwayson":
        final_selector = AlwaysOnSelector()
    else:
        print("FP Selector must be in {alwayson, threshold}")
        exit()
    return final_selector

class AlwaysOnSelector():
    def select(self, probability):
        draw = np.random.uniform(0, 1)
        return draw < probability

    def mark(self, examples):
        for example in examples:
            example.forward_select_probability = 1.
            example.forward_select = self.select(example.forward_select_probability)
        return examples

class ThresholdSelector():
    def __init__(self):
        self.logger = {"counter": 0, "path_3": 0, "path_2": 0, "path_1": 0}
        self.historical_sps = {}
        self.times_passed = {}
        self.threshold = 0.01
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
            return True
        else:
            self.times_passed[image_id] = 0
            self.logger['path_3'] += 1
            return False

    def mark(self, examples):
        for example in examples:
            example.forward_select = self.select(example)
        return examples
