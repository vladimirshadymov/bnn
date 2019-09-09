# Import PyTorch
import torch  # import main library
from torch.autograd import Function  # import Function to create custom activations
from torch import nn
from torch.nn import functional as F
import numpy as np

class BinarizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # save input for backward pass

        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input = grad_input * grad_output

        return grad_input

class StochasticBinarizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # save input for backward pass

        output = input.clone()
        output = output.add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input = grad_input * grad_output

        return grad_input

class Binarization(nn.Module):

    def __init__(self, stochastic=False):
        super(Binarization, self).__init__()
        self.stochastic = stochastic

    def forward(self, input):

        if self.stochastic:
            return StochasticBinarizeFunction.apply(input)
        else:
            return BinarizeFunction.apply(input)

def hinge_p_loss(output, target, p=0.5, reduction='sum'):

    tmp = torch.zeros_like(output)
    for i in range(tmp.shape[0]):
        tmp[i][target[i]] = 2.
    tmp -= 1.

    if reduction == 'mean':
        return torch.mean(torch.max(1.-tmp*output, torch.zeros_like(output))**p)
    elif reduction == 'mean':
        return torch.sum(torch.max(torch.zeros_like(output), 1.-tmp*output)**p)

def generate_rand_mask(kernel_shape, input_dim_x, input_dim_y, num_nonzero=16):

    mask = []
    for i in range(input_dim_x-kernel_shape[2]+1):
        tmp = []
        for j in range(input_dim_y-kernel_shape[3]+1):
            tsr = torch.ones(kernel_shape)/10
            tsr = torch.bernoulli(tsr)
            tmp.append(tsr)
        mask.append(tmp)

    return mask


class BinarizedLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data)  # clip weights #TODO: check if the inplace version better
        out = nn.functional.linear(input, BinarizeFunction.apply(self.weight), bias=self.bias)  # linear layer with binarized weights
        return out

class BinarizedConv2d(nn.Conv2d):

    def __init__(self, pruning=None, *kargs, **kwargs):
        super(BinarizedConv2d, self).__init__(*kargs, **kwargs)
        self.pruning = pruning
        self.weight_mask = None
        if pruning=='rand':
            self.weight_mask = generate_rand_mask(self.weight.shape, 34, 34)

    def forward(self, input):
        if self.pruning is None:
            self.weight.data = nn.functional.hardtanh_(self.weight.data)
            out = nn.functional.conv2d(input, BinarizeFunction.apply(self.weight), self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        elif self.pruning == 'rand':
            out = torch.zeros_like(input)
            #todo: generte tsr with right dims

        return out