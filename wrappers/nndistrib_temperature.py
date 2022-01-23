import torch


from .nn_distrib import DistributionModule
from .scheduler_parameter import regular_scheduler



class DistributionWithTemperatureParameter(DistributionModule):
    def __init__(self, distribution, temperature_train = 0.5, scheduler_parameter = regular_scheduler, test_temperature = 0.0, **kwargs):
        super(DistributionWithTemperatureParameter, self).__init__(distribution,)
        self.current_distribution = None
        self.temperature = torch.tensor(temperature_train, dtype=torch.float32)
        self.test_temperature = test_temperature
        self.scheduler_parameter = scheduler_parameter


    def forward(self, probs):
        if self.training :
            self.current_distribution = self.distribution(probs =probs, temperature = self.temperature)
        else :
            self.current_distribution = self.distribution(probs =probs, temperature = self.test_temperature)
        return self.current_distribution


    def sample_function(self, sample_shape):
        if self.training :
            return self.current_distribution.rsample(sample_shape)
        else :
            return self.current_distribution.sample(sample_shape)

    def update_distribution(self, epoch = None):
        if self.scheduler_parameter is not None :
            self.temperature = self.scheduler_parameter(self.temperature, epoch)

