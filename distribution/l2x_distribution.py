
import torch
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.utils import clamp_probs, broadcast_all
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property

from .utils import  argmax_STE

import numpy as np




### L2X Distribution and Relaxed


class L2XDistribution(torch.distributions.RelaxedOneHotCategorical):
  def __init__(self, temperature=1, probs = None, logits = None, k=None, validate_args=None):
        super(L2XDistribution, self).__init__(temperature, probs, logits, validate_args)
        self.k = k


  def rsample(self, n_samples):
        samples = super(L2XDistribution, self).rsample(n_samples).unsqueeze(0)
        for k in range(self.k-1):
            samples = torch.cat((samples, super(L2XDistribution, self).rsample(n_samples).unsqueeze(0)), 0)
        samples = torch.max(samples, dim=0)
        return samples      



## L2X Distribution STE :


class L2XDistribution_STE(torch.distributions.RelaxedOneHotCategorical):
    def __init__(self, temperature=1, probs = None, logits = None, k=None, validate_args=None):
        super(L2XDistribution_STE, self).__init__(temperature, probs, logits, validate_args)
        self.k = k

    def rsample(self, n_samples):
        samples = super().rsample(n_samples).unsqueeze(0)
        samples = argmax_STE.apply(samples)
        return samples      

    def log_prob(self, value):
        raise NotImplementedError()

    def prob(self, value):
        raise NotImplementedError()

