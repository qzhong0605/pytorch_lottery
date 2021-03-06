import torch
import torch.nn as nn
import torch.nn.functional as F
from models import base_model

class FC(base_model.HookModule):
    def __init__(self, device, name):
        super(FC, self).__init__(device, name)
        self.features = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28*28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self,x):
        out = self.features(x)
        return out


def build_fc(device):
    return FC(device, 'fc').to(device)
