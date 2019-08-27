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
    def select(self, probability):
        draw = np.random.uniform(0, 1)
        return draw < probability

    def mark(self, examples):
        for example in examples:
            example.forward_select_probability = 1.
            example.forward_select = self.select(example.forward_select_probability)
        return examples
