
from gc import unfreeze
import torch
from torch.distributions import constraints
from torch.distributions.utils import clamp_probs
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution


from scipy.special import comb
from .khot_subset_sampling import KHotSubsetSampling
from .transforms import topKSTE, threshold_STE, topKcontinuous

class ExpRelaxdeSubsetSampling(Distribution):
    r"""
    Creates a ExpRelaxedSubsetSampling parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`ExpRelaxdeSubsetSampling`.

    Implementation based on [1].

    See also: :func:`torch.distributions.ExpRelaxdeSubsetSampling`

    Args:
        temperature (Tensor): relaxation temperature
        k (int): number of classes
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    support = constraints.real_vector  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(self, temperature, k, probs=None, logits=None, validate_args=None):
        self._khot_subset_sampling = KHotSubsetSampling(k = k, probs = probs, logits = logits, validate_args = validate_args)
        self._k = k
        self.temperature = temperature
        batch_shape = self._khot_subset_sampling.batch_shape
        event_shape = self._khot_subset_sampling.param_shape[-1:]
        super(ExpRelaxdeSubsetSampling, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExpRelaxdeSubsetSampling, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._khot_subset_sampling = self._khot_subset_sampling.expand(batch_shape)
        super(ExpRelaxdeSubsetSampling, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._khot_subset_sampling._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._khot_subset_sampling.param_shape

    @property
    def logits(self):
        return self._khot_subset_sampling.logits

    @property
    def probs(self):
        return self._khot_subset_sampling.probs

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device))
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels)/self.temperature
        return scores - scores.logsumexp(dim=-1, keepdim=True) + torch.log(torch.tensor(self._k, dtype =torch.float32)) 

    def log_prob(self, value):
        raise NotImplementedError
        # K = self._khot_subset_sampling._num_events
        # if self._validate_args:
        #     self._validate_sample(value)
        # logits, value = broadcast_all(self.logits, value)
        # log_scale = (torch.full_like(self.temperature, float(K)).lgamma() -
        #              self.temperature.log().mul(-(K - 1)))
        # score = logits - value.mul(self.temperature)
        # score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        # return score + log_scale


class RelaxedSubsetSampling(TransformedDistribution):
    r"""
    Creates a RelaxedSubsetSampling distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.

    Example::

        # >>> m = RelaxedSubsetSampling(torch.tensor([2.2]),
        #                                  torch.tensor([0.1, 0.2, 0.3, 0.4]))
        # >>> m.sample()
        # tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        k (int): number of features to sample
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    # support = constraints.simplex
    has_rsample = True

    def __init__(self, temperature, k, probs=None, logits=None, validate_args=None):
        base_dist = ExpRelaxdeSubsetSampling(temperature = temperature, k = k, probs = probs, logits = logits, validate_args=validate_args)
        self._k = k
        self._temperature = temperature
        super(RelaxedSubsetSampling, self).__init__(base_dist,
                                                    topKcontinuous(k = self._k, temperature=self._temperature),
                                                    validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedSubsetSampling, _instance)
        return super(RelaxedSubsetSampling, self).expand(batch_shape, _instance=new)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs


class RelaxedSubsetSampling_STE(TransformedDistribution):
    r"""
    Creates a RelaxedSubsetSampling distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.

    Example::

        # >>> m = RelaxedSubsetSampling(torch.tensor([2.2]),
        #                                  torch.tensor([0.1, 0.2, 0.3, 0.4]))
        # >>> m.sample()
        # tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        k (int): number of features to sample
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    # support = constraints.simplex
    has_rsample = True

    def __init__(self, temperature, k, probs=None, logits=None, validate_args=None):
        base_dist = ExpRelaxdeSubsetSampling(temperature, k, probs, logits, validate_args=validate_args)
        self._k = k
        super(RelaxedSubsetSampling_STE, self).__init__(base_dist,
                                                    topKSTE(k = self._k),
                                                    validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedSubsetSampling_STE, _instance)
        return super(RelaxedSubsetSampling_STE, self).expand(batch_shape, _instance=new)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs
