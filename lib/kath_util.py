import numpy as np

class Condition(object):
    """An interface for use with the ConditionalStartSampler."""
    @property
    def satisfied(self):
        raise NotImplementedError()

    @property
    def previously_satisfied(self):
        pass  # not necessary

    def update(self, scores):
        pass  # not necessary

class VarianceReductionCondition(Condition):
    """Sample with importance sampling when the variance reduction is larger
    than a threshold. The variance reduction units are in batch size increment.
    Arguments
    ---------
        vr_th: float
               When vr > vr_th start importance sampling
        momentum: float
                  The momentum to compute the exponential moving average of
                  vr
    """
    def __init__(self, vr_th=1.2, momentum=0.9):
        #tau_th = float(B + 3*b) / (3*b)
        self._vr_th = vr_th
        self._vr = 0.0
        self._previous_vr = 0.0
        self._momentum = momentum

    @property
    def variance_reduction(self):
        return self._vr

    @property
    def satisfied(self):
        print(self._vr, self._vr_th)
        self._previous_vr = self._vr
        return self._vr > self._vr_th

    @property
    def previously_satisfied(self):
        return self._previous_vr > self._vr_th

    def update(self, scores):
        scores = np.array(scores)
        u = 1.0/len(scores)
        S = scores.sum()
        if S == 0:
            g = np.array(u)
        else:
            g = scores/S
        new_vr = 1.0 / np.sqrt(1 - ((g-u)**2).sum()/(g**2).sum())
        self._vr = (
            self._momentum * self._vr +
            (1-self._momentum) * new_vr
        )
