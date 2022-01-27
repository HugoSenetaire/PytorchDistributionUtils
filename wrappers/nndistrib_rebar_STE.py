import torch


from torch.distributions import Bernoulli, RelaxedBernoulli
from ..distribution import threshold_STE

from .utils import reparam_pz, reparam_pz_b, sigma_lambda, Heaviside
from .nn_distrib import DistributionModule


class REBARBernoulli_STE(DistributionModule):
    def __init__(self, temperature_init = 1.0, **kwargs):
        super(REBARBernoulli_STE, self).__init__(distribution = Bernoulli,)
        self.temperature = torch.nn.Parameter(torch.tensor(temperature_init), requires_grad = False)
        self.distribution_relaxed = RelaxedBernoulli

    def forward(self, probs):
        self.current_distribution = self.distribution(probs = probs,)
        self.current_distribution_relaxed = self.distribution_relaxed(probs = probs, temperature = self.temperature )
        self.distribution_parameters = probs
        return self.current_distribution, self.current_distribution_relaxed

    def log_prob(self, z):
        return self.current_distribution.log_prob(z)

    def sample(self, sample_shape = (1,)):
        if self.training :
            shape_distribution_parameters = self.distribution_parameters.shape
            complete_size = torch.Size(sample_shape) + shape_distribution_parameters 


            pi_list = self.distribution_parameters
            wanted_device = self.distribution_parameters.device

            u = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)
            v_p = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)
            z = reparam_pz(u, pi_list)
            s = Heaviside(z)
            z_tilde = reparam_pz_b(v_p, s, pi_list)
            sig_z = sigma_lambda(z, self.temperature)
            sig_z_tilde = sigma_lambda(z_tilde, self.temperature)

            sig_z = threshold_STE.apply(sig_z)
            sig_z_tilde = threshold_STE.apply(sig_z_tilde)

            return [sig_z, s, sig_z_tilde]
        else :
            
            return self.current_distribution.sample(sample_shape)