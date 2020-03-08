import torch.nn as nn

__atomic_model__ = [
        nn.Conv1d, nn.Conv2d, nn.Conv3d,   # convolution module
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, # transpose convolution module
        nn.ReLU, nn.RReLU, nn.ReLU6, nn.LeakyReLU, nn.Sigmoid, # activation module
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, # average pool module
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, # max pool module
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, # adaptive average pool module
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, # adaptive max pool module
        nn.Linear, # fully-connected module
        nn.LogSoftmax, # classifier module
        nn.Flatten,  # flatten module
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, # batch normalization module
]

def is_atomic_module(module):
    for __module__ in __atomic_model__:
        if isinstance(module, __module__):
            return True
    return False
