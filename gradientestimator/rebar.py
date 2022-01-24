from .gradient_estimator import GradientMonteCarloEstimator

import torch



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
        log_prob = torch.sum(p_z.log_prob(s).flatten(2), axis=-1)
        
        p_f_s = f(s.detach(), )
        p_f_sig_z = f(sig_z, )
        p_f_sig_z_tilde = f(sig_z_tilde, )


        loss_f = torch.mean(p_f_s.clone(), axis=0)


        reward = p_f_s.detach() - p_f_sig_z_tilde.detach()
        reward = reward * log_prob

        E_0 = p_f_sig_z - p_f_sig_z_tilde

        loss_z = torch.mean(reward + E_0, axis = 0)

        return loss_z, loss_f
        



