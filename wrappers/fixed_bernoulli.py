from .nn_distrib import DistributionModule

import torch




#### Distribution Module

class FixedBernoulli(DistributionModule):
    def __init__(self, value = 0.5, **kwargs):
        super(FixedBernoulli, self).__init__(distribution = torch.distributions.Bernoulli, **kwargs)
        self.value = value
    
    def forward(self, distribution_parameters, ):
        self.current_distribution = self.distribution(probs = torch.full_like(distribution_parameters, self.value))
        return self.current_distribution
        
    def sample_function(self, sample_shape):
        return self.current_distribution.sample(sample_shape)

