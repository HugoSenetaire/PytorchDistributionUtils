import torch
import numpy as np

# TODO : https://pytorch.org/docs/stable/notes/extending.html#extending-autograd Reread everything and make grad checks

EPSILON = torch.tensor(np.finfo(float).eps)
def continuous_topk(scores, k, temperature, separate=False):
        """
        Returns the top-k samples from the distribution.
        Args:
            scores (Tensor): the logits
            k (int): the number of samples to return
            temperature (Tensor): the temperature
            separate (bool): whether to return the top-k samples separately
        Returns:
            Tensor: the top-k samples
        """

        khot_list = torch.zeros_like(scores).unsqueeze(0)
        onehot_approx = torch.zeros_like(scores, dtype=torch.float32)
        for i in range(k):
            khot_mask = torch.maximum(1.0 - onehot_approx, EPSILON)
            scores += torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / temperature, dim=-1)
            khot_list = torch.cat([khot_list, onehot_approx.unsqueeze(0)], dim=0)
        if separate:
            return khot_list[1:]
        else:
            return khot_list[1:].sum(dim=0)


### TOP K Distribution The following is not correct
class topK_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        # ctx.save_for_backward(input, k)
        print("TOPK STE is not realy working because of the dimension. Be careful when using it.")
        _, subset_size_indices = input.topk(k, dim=-1, largest=True, sorted=False)
        if input.is_cuda:
            subset_size_indices = subset_size_indices.cuda()
            output = torch.zeros_like(input, dtype=input.dtype).scatter_(-1, subset_size_indices, torch.ones_like(input))
        else :
            output = torch.zeros(input.shape, dtype=input.dtype).scatter_(-1, subset_size_indices, 1.0)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None



### Argmax straight through


class argmax_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        print("Argmax STE is not safe because of the choice of the dimension. TODO: fix this") #TODO
        index = torch.argmax(input, dim=-1, keepdim=True)
        aux = torch.zeros_like(input).scatter_(-1, index, torch.ones(input.shape, dtype=input.dtype))
        return torch.clamp(torch.sum(aux, dim=0), min=0, max=1) # Clamp is needed to get one-hot vector

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output
        return grad_output, None

        
### Threshold Straight Through Estimator

class threshold_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ratio):
        # ctx.save_for_backward(input, k)
        device = input.device
        if not torch.is_tensor(ratio):
            ratio = torch.tensor(ratio, device=device)
        else :
            ratio.to(device)
        return torch.where(input > ratio, torch.ones(input.shape, dtype=input.dtype, device=device), torch.zeros(input.shape, dtype=input.dtype, device=device))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

