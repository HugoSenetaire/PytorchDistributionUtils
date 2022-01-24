import torch.nn as nn



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
        self.combined_grad_f_s = False
        self.fix_n_mc = False


    def __call__(self, f, param_distribution, n_mc = 1,):
        """
        :param f: The function to deal with z. Must return something of the shape : [n_mc, batch_size,]
        :param param_distribution: The parameters of the distribution, usually, we want to backward through 
        :param n_mc: The number of Monte Carlo samples for the gradient estimation.
        :return: The function to backward from to get the gradient
        """
        raise(NotImplementedError)
