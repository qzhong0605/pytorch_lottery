""" It's used to test the base hook module
"""
import torchvision
import torch
import torch.nn as nn

from models import base_model

class CustomAlexnet(base_model.HookModule):
    def __init__(self, device):
        super(CustomAlexnet, self).__init__(device)
        self._alexnet = torchvision.models.alexnet()
        self._alexnet.to(device)

    def forward(self, x):
        return self._alexnet(x)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    custom_alexnet = CustomAlexnet(device)
    custom_alexnet.dump_tensor_shape((1,3,224,224))
