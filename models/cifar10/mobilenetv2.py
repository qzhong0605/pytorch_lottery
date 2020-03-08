import torch
import torch.nn as nn
import torch.nn.functional as F

from models import base_model
from lottery_modules import elements

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.features = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.elements_sum = elements.Sum()

    def forward(self, x):
        out = self.features(x)
        out = self.elements_sum((out, self.shortcut(x) if self.stride == 1 else out))
        return out


class MobileNetV2(base_model.HookModule):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, device, name, num_classes=10):
        super(MobileNetV2, self).__init__(device, name)

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layers = self._make_layers(in_planes=32)

        self.postprocess = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(start_dim=1),
            nn.Linear(1280, num_classes)
        )

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.preprocess(x)
        out = self.layers(out)
        out = self.postprocess(out)

        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = self.classifier(out)
        return F.log_softmax(out)


def build_mobilenetv2(device):
    return MobileNetV2(device, 'mobilenetv2').to(device)
