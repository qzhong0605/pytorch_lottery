import torch
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F

from models import base_model

class LeNet(base_model.HookModule):
    def __init__(self, device, num_classes=10):
        super(LeNet, self).__init__(device)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*14*14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        output = F.log_softmax(x)
        return output

def build_lenet(device):
    return LeNet(device).to(device)
