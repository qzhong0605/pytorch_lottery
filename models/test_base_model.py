""" It's used to test the base hook module
"""
import torchvision
import torch.nn as nn

from models import base_model

class CustomAlexnet(base_model.HookModule):
    def __init__(self):
        super(CustomAlexnet, self).__init__()
        self._alexnet = torchvision.models.alexnet()

    def forward(self, x):
        return self._alexnet(x)


if __name__ == '__main__':
    custom_alexnet = CustomAlexnet()
    custom_alexnet.dump_tensor_shape((1,3,224,224))
