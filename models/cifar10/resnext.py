'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import base_model
from pytorch_lottery.lottery_modules import elements


class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.process = nn.Sequential(
            nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False),  # conv1
            nn.BatchNorm2d(group_width),  # bn1
            nn.ReLU(),  # relu1
            nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False), # conv2
            nn.BatchNorm2d(group_width),  # bn2
            nn.ReLU(),  # relu2
            nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False),  # conv3
            nn.BatchNorm2d(self.expansion*group_width)  # bn3
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )
        self.sum_out = elements.Sum()
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.sum_out((self.process(x), self.shortcut(x)))
        out = self.relu_out(out)
        return out


class ResNeXt(base_model.HookModule):
    def __init__(self, num_blocks, cardinality, bottleneck_width, device, name, num_classes=10):
        super(ResNeXt, self).__init__(device, name)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(start_dim=1),
            nn.Linear(cardinality*bottleneck_width*8, num_classes)
        )

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.preprocess(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.classifier(out)
        return out


def build_resnext29_2x64d(device):
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64,
                   device, 'resnext29_2x64d').to(device)

def build_resnext29_4x64d(device):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64,
                   device, 'resnext29_4x64d').to(device)

def build_resnext29_8x64d(device):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64,
                   device, 'resnext29_8x64d').to(device)

def build_resnext29_32x4d(device):
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4,
                   device, 'resnext29_32x64d').to(device)
