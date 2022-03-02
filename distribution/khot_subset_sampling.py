
from gc import unfreeze
import torch
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.utils import clamp_probs, broadcast_all
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property

from .utils import topK_STE, argmax_STE, threshold_STE, continuous_topk
from torch.distributions.one_hot_categorical import OneHotCategorical

from scipy.special import comb
from .subset_sampling import SubsetSampling


class KHotSubsetSampling(Distribution):
    r"""
    Creates a subset sampling distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are k-hot coded vectors of size ``probs.size(-1)``.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        # >>> m = SubsetSampling(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        # >>> m.sample()  # equal probability of 0, 1, 2, 3
        # tensor([ 0.,  0.,  0.,  1.])

    Args:
        k (int): number of classes
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    support = constraints.one_hot
    has_enumerate_support = True

    def __init__(self, k, probs=None, logits=None, validate_args=None):
        self._subset_sampling = SubsetSampling(k = k, probs = probs, logits = logits, validate_args=validate_args)
        batch_shape = self._subset_sampling.batch_shape
        event_shape = self._subset_sampling.param_shape[-1:]
        super(KHotSubsetSampling, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(KHotSubsetSampling, _instance)
        batch_shape = torch.Size(batch_shape)
        new._subset_sampling = self._subset_sampling.expand(batch_shape)
        super(KHotSubsetSampling, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._subset_sampling._new(*args, **kwargs)

    @property
    def _param(self):
        return self._subset_sampling._param

    @property
    def probs(self):
        return self._subset_sampling.probs # This is a bit weird, we do not have the probabilities exactly here since it is defined on features and not on subset...

    @property
    def logits(self):
        return self._subset_sampling.logits # This is more of a score per feature.

    @property
    def mean(self):
        return self._subset_sampling.probs * self._k # The mean of the RelaxedSubsetSampling distribution is the mode of the underlying Categorical distribution.

    @property
    def variance(self):
        raise NotImplementedError # TODO @ hhjs, it should not be that complicated
        # return self._subset_sampling.probs * (1 - self._subset_sampling.probs)

    @property
    def param_shape(self):
        return self._subset_sampling.param_shape

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        probs = self._subset_sampling.probs
        num_events = self._subset_sampling._num_events
        indices = self._subset_sampling.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).to(probs)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # indices = value.max(-1)[1]
        indices = torch.nonzero(value,)
        return self._subset_sampling.log_prob(indices)

    def entropy(self):
        return self._subset_sampling.entropy()

    def enumerate_support(self, expand=True):
        raise NotImplementedError
        # n = self.event_shape[0]
        # values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)
        # values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        # if expand:
        #     values = values.expand((n,) + self.batch_shape + (n,))
        # return values

