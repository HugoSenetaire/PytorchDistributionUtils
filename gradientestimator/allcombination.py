from .gradient_estimator import GradientMonteCarloEstimator
from .utils import get_all_z

import numpy as np
import torch



class AllCombination(GradientMonteCarloEstimator):
    """
    Exact calculation for the expectation of the discrete variable
    """

    def __init__(self, distribution, ):
        super(AllCombination, self).__init__(distribution,)
        self.all_z = None
        self.nbdim = None
        self.fix_n_mc = True

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
        log_prob = torch.sum(p_z.log_prob(z).flatten(2),axis=-1)
        output = f(z)
        
        loss_z = torch.sum(output * torch.exp(log_prob), axis=0)
        loss_f = loss_z.clone()
        return loss_z, loss_f




