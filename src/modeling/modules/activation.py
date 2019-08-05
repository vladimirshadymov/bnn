# Import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict

# Import PyTorch
import torch # import main library
from torch.autograd import Function # import Function to create custom activations


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
        grad_input[torch.abs(input) >= 1.] = 0.
        grad_input[torch.abs(input) < 1.] = 1.
        grad_input = grad_input * grad_output

        return grad_input



