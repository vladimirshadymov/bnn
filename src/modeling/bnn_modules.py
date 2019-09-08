# Import PyTorch
import torch  # import main library
from torch.autograd import Function  # import Function to create custom activations
from torch import nn


class SignFunction(Function):

    @staticmethod
    def forward(ctx, input, stochastic=False):
        ctx.save_for_backward(input)  # save input for backward pass

        if not stochastic:
            output = torch.sign(input)
        else:
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

class SignBlock(nn.Module):

    def __init__(self):
        


class BinarizedLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data)  # clip weights #TODO: check if the inplace version better
        out = nn.functional.linear(input, SignFunction.apply(self.weight), bias=self.bias)  # linear layer with binarized weights
        return out

class BinarizedConv2d(nn.Conv2d):

    def __init__(self, pruning=None, *kargs, **kwargs):
        super(BinarizedConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data)
        out = nn.functional.conv2d(input, SignFunction.apply(self.weight), self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        return out