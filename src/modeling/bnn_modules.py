# Import PyTorch
import torch  # import main library
from torch.autograd import Function  # import Function to create custom activations
from torch import nn


class SignEst(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # save input for backward pass

        # clone the input tensor
        output = input.clone()
        output[output >= 0] = 1.
        output[output < 0] = -1.

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input = grad_input * grad_output

        return grad_input


class BinarizedLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = nn.functional.linear(input, SignEst.apply(self.weight), bias=SignEst.apply(self.bias))  # linear layer with binarized weights
        self.weight.data = nn.functional.hardtanh(self.weight.data)  # clip weights #TODO: check if the inplace version better
        return out


class BinarizedConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizedConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = nn.functional.conv2d(input, SignEst.apply(self.weight), SignEst.apply(self.bias), self.stride,
                                   self.padding, self.dilation, self.groups)
        self.weight.data = nn.functional.hardtanh(self.weight.data)
        return out