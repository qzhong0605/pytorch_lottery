import torch
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F

from models import base_model

class LeNetCaffe(base_model.HookModule):
    def __init__(self, device, name, num_classes=10):
        super(LeNetCaffe, self).__init__(device, name)
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.classifier = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

def build_lenetcaffe(device):
    return LeNetCaffe(device, 'lenetcaffe').to(device)
