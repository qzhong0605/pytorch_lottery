""" MobileNet network archtecture

It's based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import base_model

class Block(nn.Module):
    """ depthwise convolution and normal convolution"""
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.features(x)
        return out


class MobileNet(base_model.HookModule):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, device, name, num_classes=10):
        super(MobileNet, self).__init__(device, name)
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layers = self._make_layers(in_planes=32)

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(1024, num_classes)
        )

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.preprocess(x)
        out = self.layers(out)
        out = self.classifier(out)
        return F.log_softmax(out)

def build_mobilenet(device):
    return MobileNet(device, 'mobilenet').to(device)
