
import torch
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.utils import clamp_probs, broadcast_all
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property

from .utils import topK_STE, argmax_STE, threshold_STE, continuous_topk

import numpy as np






## Relaxed Bernoulli thresholded STE :

class RelaxedBernoulli_thresholded_STE(torch.distributions.RelaxedBernoulli):
    def __init__(self, temperature = 1, probs =None, logits = None, threshold = 0.5, validate_args = None) -> None:
        super(RelaxedBernoulli_thresholded_STE, self).__init__(temperature, probs, logits, validate_args)
        self.threshold = threshold

    def rsample(self, n_sample):
        samples = super(RelaxedBernoulli_thresholded_STE, self).rsample(n_sample)
        samples = threshold_STE.apply(samples, self.threshold)
        return samples


