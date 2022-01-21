
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, RelaxedBernoulli

class RebarTypeDistribution(nn.Module):
    def __init__(self, distribution, distribution_relaxed):
        super().__init__()
        self.distribution = distribution
        self.distribution_relaxed = distribution_relaxed

    def forward(self, probs):
        raise(NotImplementedError)

    def sample(self, n_mc = 1):
        raise(NotImplementedError)

    def log_prob(self, z):
        raise(NotImplementedError)



# BERNOULLI REBAR :

def safe_log_prob(x, eps=1e-8):
    return torch.log(torch.clamp(x, eps, 1.0))    

def sigma_lambda(z, lambda_value):
  return torch.sigmoid(z / lambda_value)

def binary_log_likelihood(y, log_y_hat):
    return (y * -F.softplus(-log_y_hat)) + (1 - y) * (-log_y_hat - F.softplus(-log_y_hat))

def Heaviside(x):
    return torch.heaviside(x.detach(), torch.tensor(0., device = x.device))

def reparam_pz(u, pi_list):
    return (safe_log_prob(pi_list) - safe_log_prob(1 - pi_list)) + (safe_log_prob(u) - safe_log_prob(1 - u))


def reparam_pz_b(v, b, theta):
    return(b * F.softplus(safe_log_prob(v) - safe_log_prob((1 - v) * (1 - theta)))) \
        + ((1 - b) * (-F.softplus(safe_log_prob(v) - safe_log_prob(v * (1 - theta)))))





class REBARBernoulli(RebarTypeDistribution):
    def __init__(self, temperature_init = 1.0,):
        super(REBARBernoulli, self).__init__(distribution = Bernoulli, distribution_relaxed = RelaxedBernoulli,)
        self.temperature_total = torch.nn.Parameter(torch.tensor(temperature_init), requires_grad = False)

    def forward(self, probs):
        self.current_distribution = self.distribution(probs = probs,)
        self.current_distribution_relaxed = self.distribution_relaxed(probs = probs, temperature = self.temperature_total )
        self.distribution_parameters = probs
        return self.current_distribution, self.current_distribution_relaxed

    def log_prob(self, z):
        return self.current_distribution.log_prob(z)

    def sample(self, n_mc = 1):
            complete_size = torch.Size(n_mc,) + self.distribution_parameters.shape
            pi_list = self.distribution_parameters
            wanted_device = self.distribution_parameters.device
            u = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)
            v_p = (torch.rand(complete_size, requires_grad = False, device= wanted_device) + 1e-9).clamp(1e-8,1)
            z = reparam_pz(u, pi_list)
            s = Heaviside(z)
            z_tilde = reparam_pz_b(v_p, s, pi_list)
            sig_z = sigma_lambda(z, self.temperature_total)
            sig_z_tilde = sigma_lambda(z_tilde, self.temperature_total)

            return [sig_z, s, sig_z_tilde]