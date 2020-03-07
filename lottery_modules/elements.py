import torch
import torch.nn as nn

class Sum(nn.Module):
    r"""
    Sum two tensors into a output tensor. It's performed by elemenet-wise.

    The inputs must be the same shape. And the out shape is also the same with the input tensor
    """

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, input):
        """ The input must be a list with length of 2
        """
        return input[0] + input[1]
