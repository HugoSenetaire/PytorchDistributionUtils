from .gradient_estimator import GradientMonteCarloEstimator

import torch

        
class PathWise(GradientMonteCarloEstimator):
    """
    PathWise Gradient Monte Carlo estimator
    """

    def __init__(self, distribution, ):
        super(PathWise, self).__init__(distribution,)
        self.combined_grad_f_s = True

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




