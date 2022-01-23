
import torch
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.utils import clamp_probs, broadcast_all
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property

from .utils import topK_STE, argmax_STE, threshold_STE, continuous_topk

import numpy as np

class SubsetSampling(Distribution):
    r"""
    Creates a SubsetSampling distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None, subset_size = None, validate_args=None):
        if subset_size is None:
            raise ValueError("subset_size must be specified")
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)

        self.subset_size = subset_size
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size(-1) # TODO Change this to get the right number of possible event, this is definitely wrong
        batch_shape = self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        super(SubsetSampling, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SubsetSampling, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size(self.logits.shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_events = self._num_events
        super(SubsetSampling, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)


    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)


    @property
    def param_shape(self):
        return self._param.size()

    @property
    def mean(self):
        return torch.full(self._extended_shape(), nan, dtype=self.probs.dtype, device=self.probs.device)

    @property
    def variance(self):
        return torch.full(self._extended_shape(), nan, dtype=self.probs.dtype, device=self.probs.device)

    def sample(self, sample_shape=torch.Size()):
        # sample_shape = self.expand(sample_shape)
        logits = self.expand(sample_shape).logits
        uniforms = clamp_probs(torch.rand(logits.shape, dtype=logits.dtype, device=logits.device))
        scores = torch.log(uniforms)/logits
        topk_values, topk_indices = torch.topk(scores, self.subset_size, dim=-1)
        return torch.zeros_like(scores).scatter_(-1, topk_indices, 1)


        


    def log_prob(self, value):
        raise NotImplementedError()
        # if self._validate_args:
        #     self._validate_sample(value)
        # value = value.long().unsqueeze(-1)
        # value, log_pmf = torch.broadcast_tensors(value, self.logits)
        # value = value[..., :1]
        # return log_pmf.gather(-1, value).squeeze(-1)


    def entropy(self):
        raise NotImplementedError()
        # min_real = torch.finfo(self.logits.dtype).min
        # logits = torch.clamp(self.logits, min=min_real)
        # p_log_p = logits * self.probs
        # return -p_log_p.sum(-1)


    def enumerate_support(self, expand=True):
        raise NotImplementedError()
        # num_events = self._num_events
        # values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        # values = values.view((-1,) + (1,) * len(self._batch_shape))
        # if expand:
        #     values = values.expand((-1,) + self._batch_shape)
        # return values



class RelaxedSubsetSampling(Distribution):
    r"""
    Creates a RelaxedSubsetSampling parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`SubsetSampling`.

    Implementation based on [1].

    See also: :func:`torch.distributions.SubsetSampling`

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al, 2017)

    [2] SubsetSampling Reparametrization with Gumbel-Softmax
    (Jang et al, 2017)
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    support = constraints.real_vector  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, subset_size = None, validate_args=None):
        self._SubsetSampling = SubsetSampling(probs = probs, logits = logits, subset_size = subset_size)
        self.temperature = temperature
        self.subset_size = subset_size
        batch_shape = self._SubsetSampling.batch_shape
        event_shape = self._SubsetSampling.param_shape[-1:]
        super(RelaxedSubsetSampling, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedSubsetSampling, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._SubsetSampling = self._SubsetSampling.expand(batch_shape)
        super(RelaxedSubsetSampling, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._SubsetSampling._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._SubsetSampling.param_shape

    @property
    def logits(self):
        return self._SubsetSampling.logits

    @property
    def probs(self):
        return self._SubsetSampling.probs

   

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device))
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels)
        sample = continuous_topk(scores, self.subset_size, self.temperature, separate=False)
        return sample

    def log_prob(self, value):
        raise NotImplementedError("Need to implement this for the subset sampling, can only be used as a sampler right now")
        # K = self._SubsetSampling._num_events
        # if self._validate_args:
        #     self._validate_sample(value)
        # logits, value = broadcast_all(self.logits, value)
        # log_scale = (torch.full_like(self.temperature, float(K)).lgamma() -
        #              self.temperature.log().mul(-(K - 1)))
        # score = logits - value.mul(self.temperature)
        # score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        # return score + log_scale



class RelaxedSubsetSampling_STE(RelaxedSubsetSampling):
    def __init__(self, temperature, probs=None, logits=None, subset_size = None, validate_args=None):
        super(RelaxedSubsetSampling_STE, self).__init__(temperature, probs, logits, subset_size, validate_args)

    def rsample(self, sample_shape=torch.Size()):
        sample = super().rsample(sample_shape=sample_shape)
        return topK_STE.apply(sample, self.subset_size)


### L2X Distribution and Relaxed


class L2X_Distribution(torch.distributions.RelaxedOneHotCategorical):
  def __init__(self, temperature=1, probs = None, logits = None, subset_size=None, validate_args=None):
        super(L2X_Distribution, self).__init__(temperature, probs, logits, validate_args)
        self.subset_size = subset_size


  def rsample(self, n_samples):
        samples = super(L2X_Distribution, self).rsample(n_samples).unsqueeze(0)
        for k in range(self.subset_size-1):
            samples = torch.cat((samples, super(L2X_Distribution, self).rsample(n_samples).unsqueeze(0)), 0)
        samples = torch.max(samples, dim=0)
        return samples      



## L2X Distribution STE :


class L2X_Distribution_STE(torch.distributions.RelaxedOneHotCategorical):
    def __init__(self, temperature=1, probs = None, logits = None, subset_size=None, validate_args=None):
        super(L2X_Distribution_STE, self).__init__(temperature, probs, logits, validate_args)
        self.subset_size = subset_size

    def rsample(self, n_samples):
        samples = super().rsample(n_samples).unsqueeze(0)
        samples = argmax_STE.apply(samples)
        return samples      

    def log_prob(self, value):
        raise NotImplementedError()

    def prob(self, value):
        raise NotImplementedError()

## Relaxed Bernoulli thresholded STE :


class RelaxedBernoulli_thresholded_STE(torch.distributions.RelaxedBernoulli):
    def __init__(self, temperature = 1, probs =None, logits = None, threshold = 0.5, validate_args = None) -> None:
        super(RelaxedBernoulli_thresholded_STE, self).__init__(temperature, probs, logits, validate_args)
        self.threshold = threshold

    def rsample(self, n_sample):
        samples = super(RelaxedBernoulli_thresholded_STE, self).rsample(n_sample)
        samples = threshold_STE.apply(samples, self.threshold)
        return samples


