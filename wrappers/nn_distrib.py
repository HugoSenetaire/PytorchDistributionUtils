import torch.nn as nn
import torch



#### Distribution Module

class DistributionModule(nn.Module):
    """
    This is a wrapper for the pytorch distribution. 
    Makes it easier to work with parameters such as temperature that needs to be learn or scheduled.
    """
    def __init__(self, distribution, sampling_transform = None, **kwargs):
        super().__init__()
        self.sampling_transform = sampling_transform
        self.distribution = distribution
        self.current_distribution = distribution

    def forward(self, probs,):
        self.current_distribution = self.distribution(probs = probs)
        return self.current_distribution # TODO: To mimic the way Pytorch is doing ?

    def log_prob(self, x):
        return self.current_distribution.log_prob(x)

    def sample_function(self, sample_shape):
        return self.current_distribution.sample(sample_shape)

    def rsample(self, sample_shape = (1,)):
        return self.current_distribution.rsample(sample_shape)

    def sample(self, sample_shape= (1,)):
        sample = self.sample_function(sample_shape)
        # return self.sampling_transform(sample) # TODO : How to handle some sampling transform ?
        return sample

    def update_distribution(self, epoch = None):
        return None

