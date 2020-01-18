# Import PyTorch
import torch  # import main library
from torch.autograd import Function  # import Function to create custom activations
from torch import nn
from torch.nn import functional as F
import math
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

    def __init__(self, min=-1, max=1, stochastic=False):
        super(Binarization, self).__init__()
        self.stochastic = stochastic
        self.min = min
        self.max = max

    def forward(self, input):

        if self.stochastic:
            return 0.5*(StochasticBinarizeFunction.apply(input)*(self.max-self.min) + self.min + self.max)
        else:
            return 0.5*(BinarizeFunction.apply(input)*(self.max - self.min) + self.min + self.max)

class MemBinLinear(nn.Linear):

    def __init__(self, min_value=-1, max_value=+1, min_weight=-1, max_weight=+1, active=False, *kargs, **kwargs):
        super(MemBinLinear, self).__init__(*kargs, **kwargs)
        # state of the memory layer
        self.active = active
        self.count = 0
        self.avg_tsr = 0

        # init binarized limits
        self.min_value = min_value
        self.max_value = max_value
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.binarization = Binarization(min=min_weight, max=max_weight)

        # Let w*x to be convolution product of the object x by the kernel w. $w, x \in {-1, +1}$.
        # Input voltage: V = pv*x + qv
        # Weights' conductance: C = pc*w + qc
        # Reverse transform:
        # x = av*V + bv
        # w = ac*W + bc

        # affine transformations parameters
        self.pv = 0.5*(self.max_value - self.min_value)
        self.qv = 0.5*(self.max_value + self.min_value)

        self.pc = 0.5*(self.max_weight - self.min_weight)
        self.qc = 0.5*(self.max_weight + self.min_weight)

        self.av = 1.0/self.pv
        self.bv = -self.qv/self.pv

        self.ac = 1.0/self.pc
        self.bc = -self.qc/self.pc

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data)
        tmp = nn.functional.linear(self.av*0.5*(input*self.pv + self.qv), self.bc*torch.ones_like(self.weight.data), bias=self.bias)

        out = nn.functional.linear(input, self.binarization(self.weight), bias=self.bias)

        if self.train:
            self.count += 1
            self.avg_tsr += (tmp - self.avg_tsr)/self.count
        else:
            self.count = 0
        
        if self.active:
            return (out - tmp + self.avg_tsr)
        else:
            return out

class MemBinConv2d(nn.Conv2d):

    def __init__(self, min_value=-1, max_value=+1, min_weight=-1, max_weight=+1, active=False, *kargs, **kwargs):
        super(MemBinConv2d, self).__init__(*kargs, **kwargs)
        # state of the memory layer
        self.active = active
        self.count = 0
        self.avg_tsr = 0

        # init binarized limits
        self.min_value = min_value
        self.max_value = max_value
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.binarization = Binarization(min=min_weight, max=max_weight)

        # Let w*x to be convolution product of the object x by the kernel w. $w, x \in {-1, +1}$.
        # Input voltage: V = pv*x + qv
        # Weights' conductance: C = pc*w + qc
        # Reverse transform:
        # x = av*V + bv
        # w = ac*W + bc

        # affine transformations parameters
        self.pv = 0.5*(self.max_value - self.min_value)
        self.qv = 0.5*(self.max_value + self.min_value)

        self.pc = 0.5*(self.max_weight - self.min_weight)
        self.qc = 0.5*(self.max_weight + self.min_weight)

        self.av = 1.0/self.pv
        self.bv = -self.qv/self.pv

        self.ac = 1.0/self.pc
        self.bc = -self.qc/self.pc

    def forward(self, input):

        self.weight.data = nn.functional.hardtanh_(self.weight.data)
        tmp = nn.functional.conv2d(self.av*0.5*(input*self.pv + self.qv), self.bc*torch.ones_like(self.weight.data), self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

        out = nn.functional.conv2d(input, self.binarization(self.weight), self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

        if self.train:
            self.count += 1
            self.avg_tsr += (tmp - self.avg_tsr)/self.count
        else:
            self.count = 0
        
        if self.active:
            return (out - tmp + self.avg_tsr)
        else:
            return out
