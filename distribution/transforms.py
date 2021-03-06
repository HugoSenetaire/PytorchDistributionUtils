from torch.distributions.transforms import Transform
from torch.distributions import constraints

from .utils import threshold_STE, topK_STE, continuous_topk

class ThresholdSTE(Transform):
    r"""
    Transform via the mapping :math:`y = x>self.rate`.
    Allows for gradient to flow with the STE.
    """
    domain = constraints.unit_interval
    codomain = constraints.boolean

    def __init__(self, cache_size=0, rate = 0.5,):
        super(ThresholdSTE, self).__init__(cache_size=cache_size)
        self.rate = rate

    def __eq__(self, other):
        return isinstance(other, ThresholdSTE)

    def _call(self, x):
        return threshold_STE.apply(x, rate = self.rate)

    def _inverse(self, y):
        return y # TODO : Is it better to have None ?


class topKSTE(Transform):
    """
    Transforms via the mapping :math:`y = `.
    Allows for gradient to flow with the STE.
    """
    domain = constraints.unit_interval
    codomain = constraints.boolean

    def __init__(self, cache_size=0, k = 1,):
        super(topKSTE, self).__init__(cache_size=cache_size)
        self.k = k

    def __eq__(self, other):
        return isinstance(other, topKSTE)

    def _call(self, x):
        return topK_STE.apply(x, self.k)

    def _inverse(self, y):
        return y # TODO : Is it better to have None ?

class topKcontinuous(Transform):
    """
    Transforms via the mapping :math:`y = `.
    Allows for gradient to flow with the STE.
    """
    domain = constraints.unit_interval
    codomain = constraints.real

    def __init__(self, cache_size=0, k = 1, temperature = 1.0,):
        super(topKcontinuous, self).__init__(cache_size=cache_size)
        self.k = k
        self.temperature = temperature

    def __eq__(self, other):
        return isinstance(other, topKcontinuous)

    def _call(self, x):
        return continuous_topk(x, k = self.k, temperature=self.temperature)

    def _inverse(self, y):
        return y # TODO : Is it better to have None ?


class argmaxSTE(Transform):
    """
    Transforms via the mapping :math:`y = `.
    Allows for gradient to flow with the STE.
    """
    domain = constraints.unit_interval
    codomain = constraints.boolean

    def __init__(self, cache_size=0, k = 1,):
        super(topKSTE, self).__init__(cache_size=cache_size)
        self.k = k

    def __eq__(self, other):
        return isinstance(other, topKSTE)

    def _call(self, x):
        return topK_STE.apply(x, k = self.k)

    def _inverse(self, y):
        return y # TODO : Is it better to have None ?


