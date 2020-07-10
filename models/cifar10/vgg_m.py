'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from models import base_model


cfg = {
    'VGG11m': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13m': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16m': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19m': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
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
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def adjust_modules(self, bn_mask):
        """adjust the channels according to the `bn_mask`, which is 0-1 tensor"""
        new_cfg = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                pass

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

def build_vgg11m(device):
    return VGG('VGG11m', device, 'vgg11m').to(device)

def build_vgg13m(device):
    return VGG('VGG13m', device, 'vgg13m').to(device)

def build_vgg16m(device):
    return VGG('VGG16m', device, 'vgg16m').to(device)

def build_vgg19m(device):
    return VGG('VGG19m', device, 'vgg19m').to(device)
