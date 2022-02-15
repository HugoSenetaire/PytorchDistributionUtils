from .nn_distrib import DistributionModule

import torch




#### Distribution Module

class FixedBernoulli(DistributionModule):
    def __init__(self, value = 0.5, **kwargs):
        super(FixedBernoulli, self).__init__(distribution = torch.distributions.Bernoulli,)
        self.value = value
    
    def forward(self, probs, ):
        self.current_distribution = self.distribution(probs = torch.full_like(probs, self.value))
        return self.current_distribution
        
    def sample_function(self, sample_shape):
        return self.current_distribution.sample(sample_shape)

