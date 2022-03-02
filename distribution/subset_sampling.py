
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

class SubsetSampling(Distribution):
    r"""
    Creates a SubsetSampling Distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).


    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        #TODO: Add example

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
        k = int: number of samples to draw
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    has_enumerate_support = True

    def __init__(self, k, probs=None, logits=None, validate_args=None,):
        if k <= 0 :
            raise ValueError("k must be positive")
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
            if k > probs.shape[-1]:
                raise ValueError("k must be smaller than the number of features")
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            if k > logits.shape[-1]:
                raise ValueError("k must be smaller than the number of features")
        
    
        self._param = self.probs if probs is not None else self.logits
        self._k = k
        self._num_events = comb(self._param.size()[-1], self._k)
        batch_shape = self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        super(SubsetSampling, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SubsetSampling, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
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
        raise NotImplementedError #TODO : add support for k > 1
        # return constraints.integer_interval(0, self._num_events - 1)

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
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        uniforms = clamp_probs(torch.rand(sample_shape, dtype=self.probs.dtype, device=self.probs.device))
        score = (uniforms.log() / clamp_probs(self.probs)).exp()
        sample_index = torch.topk(score, self._k, dim=-1)[1] #TODO Check that this give the correct result
        return sample_index

    def log_prob(self, value):
        """ Here this is just an approximation. """
        if self._validate_args:
            self._validate_sample(value) 
        value = value.long().unsqueeze(-1) # IN CATEGORICAL : value is a tensor of size (batch_size,) Here I have a tensor of size (batch_size, k)
        # Logits is a tensor of size (batch_size, dim_features)
        # Sauf que moi j'ai plein d'index 
        # gather : out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        # out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
        value, log_pmf = torch.broadcast_tensors(value, self.logits.unsqueeze(-2))
        value = value[..., :1]
        log_pmf = torch.sum(self.logits.gather(-1, value),dim=-1) - torch.log(torch.tensor(self._num_events, dtype = torch.float32))# La j;en ai bathc_size, k
        return log_pmf
        # return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        raise NotImplementedError
        # min_real = torch.finfo(self.logits.dtype).min
        # logits = torch.clamp(self.logits, min=min_real)
        # p_log_p = logits * self.probs
        # return -p_log_p.sum(-1)

    def enumerate_support(self, expand=True):
        raise NotImplementedError
        # num_events = self._num_events
        # values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        # values = values.view((-1,) + (1,) * len(self._batch_shape))
        # if expand:
        #     values = values.expand((-1,) + self._batch_shape)
        # return values



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
        self._subset_sampling = SubsetSampling(k, probs, logits, validate_args=validate_args)
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
        self._khot_subset_sampling = KHotSubsetSampling(k, probs, logits)
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
        return scores - scores.logsumexp(dim=-1, keepdim=True) 

    def log_prob(self, value):
        K = self._khot_subset_sampling._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = (torch.full_like(self.temperature, float(K)).lgamma() -
                     self.temperature.log().mul(-(K - 1)))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score + log_scale


class RelaxedSubsetSampling(TransformedDistribution):
    r"""
    Creates a RelaxedOneHotCategorical distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.

    Example::

        >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
                                         torch.tensor([0.1, 0.2, 0.3, 0.4]))
        >>> m.sample()
        tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        base_dist = ExpRelaxedCategorical(temperature, probs, logits, validate_args=validate_args)
        super(RelaxedOneHotCategorical, self).__init__(base_dist,
                                                       ExpTransform(),
                                                       validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
        return super(RelaxedOneHotCategorical, self).expand(batch_shape, _instance=new)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs
