
import torch
import torch.nn as nn
import torch.nn.functional as F


# BERNOULLI REBAR :

def safe_log_prob(x, eps=1e-8):
    return torch.log(torch.clamp(x, eps, 1.0))    

def sigma_lambda(z, lambda_value):
  return torch.sigmoid(z / lambda_value)


def Heaviside(x):
    return torch.heaviside(x.detach(), torch.tensor(0., device = x.device))

def reparam_pz(u, pi_list):
    return (safe_log_prob(pi_list) - safe_log_prob(1 - pi_list)) + (safe_log_prob(u) - safe_log_prob(1 - u))


def reparam_pz_b(v, b, theta):
    return(b * F.softplus(safe_log_prob(v) - safe_log_prob((1 - v) * (1 - theta)))) \
        + ((1 - b) * (-F.softplus(safe_log_prob(v) - safe_log_prob(v * (1 - theta)))))


