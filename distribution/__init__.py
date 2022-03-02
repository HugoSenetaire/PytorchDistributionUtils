from .utils import threshold_STE, topK_STE, argmax_STE, continuous_topk
from .transforms import ThresholdSTE, topKSTE, topKcontinuous
from .extra_constraints import *
from .subset_sampling import SubsetSampling
from .khot_subset_sampling import KHotSubsetSampling
from .relaxed_bernoulli_threshold_STE import RelaxedBernoulli_thresholded_STE
from .relaxed_subset_sampling import RelaxedSubsetSampling, RelaxedSubsetSampling_STE
from .l2x_distribution import L2XDistribution, L2XDistribution_STE


self_regularized_distributions = [SubsetSampling, KHotSubsetSampling, RelaxedSubsetSampling, RelaxedSubsetSampling_STE, L2XDistribution, L2XDistribution_STE]