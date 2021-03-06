import torch
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F

from models import base_model

class LeNetBN(base_model.HookModule):
    def __init__(self, device, name, num_classes=10):
        super(LeNetBN, self).__init__(device, name)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.classifier = nn.Sequential(
            nn.Linear(64*14*14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

def build_lenetbn(device):
    return LeNetBN(device, 'lenetbn').to(device)
