import torch.nn as nn
import torch

from REBAR_utils import RebarTypeDistribution, REBARBernoulli


from itertools import combinations
import numpy as np
def get_all_z(dim):
    """Get all possible combinations of z"""
    output = np.zeros((2**dim, dim))
    number = 0
    for nb_pos in range(dim+1):
        combinations_index = combinations(range(dim), r = nb_pos)
        for combination in combinations_index :
            for index in combination :
                output[number,index]=1
            number+=1
    output = torch.tensor(output, dtype=torch.float)

    return output


class GradientMonteCarloEstimator(nn.Module):
    """
    This class is used to make an estimation of MonteCarloGradient 
    """

    def __init__(self, distribution,):
        """
        :param model: The model to be used for gradient estimation.
        """
        super(GradientMonteCarloEstimator, self).__init__()
        self.distribution = distribution


    def __call__(self, f, param_distribution, n_mc = 1,):
        """
        :param f: The function to deal with z. Must return something of the shape : [n_mc, batch_size,]
        :param param_distribution: The parameters of the distribution, usually, we want to backward through 
        :param n_mc: The number of Monte Carlo samples for the gradient estimation.
        :return: The function to backward from to get the gradient
        """
        raise(NotImplementedError)


class AllCombination(GradientMonteCarloEstimator):
    """
    Exact calculation for the expectation of the discrete variable
    """

    def __init__(self, distribution, ):
        super(AllCombination, self).__init__(distribution,)
        self.all_z = None
        self.nbdim = None

    def __call__(self, f, param_distribution, n_mc = 1,):
        """
        :param f: The function to deal with z. Must return something of the shape : [n_mc, batch_size,]
        :param param_distribution: The parameters of the distribution, usually, we want to backward through 
        :param n_mc: The number of Monte Carlo samples for the gradient estimation.
        :return: The function to backward from to get the gradient

        Note: Here n_mc is actually not useful, because we are not doing any Monte Carlo sampling but just using all combination from the data
        """
        p_z = self.distribution(probs = param_distribution)
        # Param distribution must be in the shape Batch_size x dim
        if self.all_z is None :
            self.nbdim = int(np.prod(param_distribution.shape[1:]))
            self.all_z = torch.tensor(get_all_z(self.nbdim), dtype=torch.float32)
            self.n_mc = self.all_z.shape[0] # 2 ** nbdim
        
        z = self.all_z.reshape(torch.Size((self.n_mc,)) + param_distribution.shape[1:])
        z = z.unsqueeze(1).expand(torch.Size((self.n_mc,)) + param_distribution.shape)
        log_prob = torch.sum(p_z.log_prob(z),axis=-1)
        output = f(z)
        
        loss_z = torch.sum(output * torch.exp(log_prob), axis=0)
        loss_f = loss_z.clone()
        return loss_z, loss_f





class REINFORCE(GradientMonteCarloEstimator):
    """
    REINFORCE estimator for Monte Carlo Gradient.
    """

    def __init__(self, distribution, ):
        super(REINFORCE, self).__init__( distribution,)


    def __call__(self, f, param_distribution, n_mc = 1,):
        """
        :param f: The function to deal with z. Must return something of the shape : [n_mc, batch_size,]
        :param param_distribution: The parameters of the distribution, usually, we want to backward through 
        :param n_mc: The number of Monte Carlo samples for the gradient estimation.
        :return: The function to backward from to get the gradient
        """
        p_z = self.distribution(probs = param_distribution)
        z = p_z.sample((n_mc,))
        log_prob = torch.sum(p_z.log_prob(z), axis=-1)

        output = f(z,)
        loss_f = torch.mean(output.clone(), axis=0)
        loss_z = torch.mean(output.detach() * log_prob, axis = 0)

        return loss_z, loss_f


class REBAR(GradientMonteCarloEstimator):
    """
    REBAR estimator for MonteCarloGradient
    """
    def __init__(self, distribution,):

        super(REBAR, self).__init__(distribution,)

    def __call__(self, f, param_distribution, n_mc = 1,):
        """
        :param f: The function to deal with z. Must return something of the shape : [n_mc, batch_size,]
        :param param_distribution: The parameters of the distribution, usually, we want to backward through 
        :param n_mc: The number of Monte Carlo samples for the gradient estimation.
        :return: The function to backward from to get the gradient
        """
        
        p_z, p_z_relaxed = self.distribution(probs = param_distribution)
        [sig_z, s, sig_z_tilde] = self.distribution.sample((n_mc,))
        log_prob = torch.sum(p_z.log_prob(s), axis=-1)

        p_f_s = f(s.detach(), )
        p_f_sig_z = f(sig_z, )
        p_f_sig_z_tilde = f(sig_z_tilde, )


        loss_f = torch.mean(p_f_s.clone(), axis=0)


        reward = p_f_s.detach() - p_f_sig_z_tilde.detach()
        reward = reward * log_prob

        E_0 = p_f_sig_z - p_f_sig_z_tilde

        loss_z = torch.mean(reward + E_0, axis = 0)

        return loss_z, loss_f
        
class PathWise(GradientMonteCarloEstimator):
    """
    PathWise Gradient Monte Carlo estimator
    """

    def __init__(self, distribution, ):
        super(PathWise, self).__init__(distribution,)

    def __call__(self, f, param_distribution, n_mc = 1,):
        """
        :param f: The function to deal with z. Must return something of the shape : [n_mc, batch_size,]
        :param param_distribution: The parameters of the distribution, usually, we want to backward through 
        :param n_mc: The number of Monte Carlo samples for the gradient estimation.
        :return: The function to backward from to get the gradient
        """
        p_z = self.distribution(probs = param_distribution)
        z = p_z.rsample((n_mc,))
        output = f(z,)
        loss_z = torch.mean(output, axis = 0)
        loss_f = loss_z.clone()
        return loss_z, loss_f




