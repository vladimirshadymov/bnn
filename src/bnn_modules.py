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
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input[torch.abs(input) > 1.] = 0.
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

def pruning_conv2d(input, weight, bias, stride, padding, dilation, p, mask=None):
    batch_size, input_x, input_y = input.shape[0], input.shape[2], input.shape[3]
    out_channels, in_channels, kernel_size = weight.data.shape[0], weight.data.shape[1], weight.data.shape[2:]
    device = input.device
    out = F.unfold(input, kernel_size, dilation, padding, stride)
    del input
    if mask is None:
        mask = torch.ones(size=(out.shape[1], out.shape[2]))*p
        mask = torch.bernoulli(mask).to(device)
        return mask
    connection_mask = mask.repeat(batch_size, 1, 1)

    out = out*connection_mask
    out = out.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)

    output_x = math.trunc((input_x + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1)/stride[0] + 1)
    output_y = math.trunc((input_y + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1)/stride[0] + 1)

    out = out.view((batch_size, out_channels, output_x, output_y))
    if not bias is None:
        out += bias.view(1, -1).unsqueeze(2).unsqueeze(3).expand_as(out)
    del connection_mask
    return out


class BinarizedLinear(nn.Linear):

    def __init__(self, min_weight=-1, max_weight=1, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)
        self.binarization = Binarization(min=min_weight, max=max_weight)
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(self, input):
        self.weight.data = nn.functional.hardtanh_(self.weight.data) 
        out = nn.functional.linear(input, self.binarization(self.weight), bias=self.bias)  # linear layer with binarized weights
        return out

class BinarizedConv2d(nn.Conv2d):

    def __init__(self, min_weight=-1, max_weight=1, p=0, *kargs, **kwargs):
        super(BinarizedConv2d, self).__init__(*kargs, **kwargs)
        self.p = p  # probability of connection pruning
        self.mask = None  # mask for connection pruning
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.binarization = Binarization(min=min_weight, max=max_weight)

    def forward(self, input):
        if self.p == 0:
            self.weight.data = nn.functional.hardtanh_(self.weight.data)
            out = nn.functional.conv2d(input, self.binarization(self.weight), self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        else:
            if self.mask is None: self.mask = pruning_conv2d(input, self.binarization(self.weight), self.bias, self.stride,
                                   self.padding, self.dilation, p=self.p)

            out = pruning_conv2d(input, self.binarization(self.weight), self.bias, self.stride,
                                   self.padding, self.dilation, p=self.p, mask=self.mask)

        return out

class PruningConv2d(nn.Conv2d):

    def __init__(self, p=0.5, *kargs, **kwargs):
        super(PruningConv2d, self).__init__(*kargs, **kwargs)
        self.p = p  # probability of connection pruning
        self.mask = None  # mask for connection pruning

    def forward(self, input):
        if self.mask is None: self.mask = pruning_conv2d(input, self.weight, self.bias, self.stride,
                                                         self.padding, self.dilation, p=self.p)

        out = pruning_conv2d(input, self.weight, self.bias, self.stride,
                                    self.padding, self.dilation, p=self.p, mask=self.mask)

        return out

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target) #TODO: change target dims
        output[output.le(0)] = 0
        self.save_for_backward(input, target)
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
        input, target = self.saved_tensors
        output = self.margin-input.mul(target)
        output[output.le(0)] = 0
        import pdb; pdb.set_trace()
        grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input.numel())
        return grad_output,grad_output


class TransformMemory(nn.Module):
    def __init__(self, min_value=-1, max_value=+1, min_weight=-1, max_weight=+1, active=False):
        super(TransformMemory, self).__init__()
        # state of the memory layer
        self.active = active
        self.count = 0
        self.avg_tsr = 0

        # init binarized limits
        self.min_value = min_value
        self.max_value = max_value
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Let w*x to be scalar product. $w, x \in {-1, +1}$.
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
        tmp = self.bc*self.av*0.5*(input*self.pv + self.qv)        
        if not self.active:
            return input
        elif self.train:
            self.count += 1
            self.avg_tsr += (tmp - self.avg_tsr)/self.count
            return input
        elif not self.train:
            return (input - tmp + self.avg_tsr)
