from .gradient_estimator import GradientMonteCarloEstimator

import torch






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

