'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from models import base_model


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(base_model.HookModule):
    def __init__(self, vgg_name, device, name, num_classes=10):
        super(VGG, self).__init__(device, name)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

def build_vgg11(device):
    return VGG('VGG11', device, 'vgg11').to(device)

def build_vgg13(device):
    return VGG('VGG13', device, 'vgg13').to(device)

def build_vgg16(device):
    return VGG('VGG16', device, 'vgg16').to(device)

def build_vgg19(device):
    return VGG('VGG19', device, 'vgg19').to(device)
