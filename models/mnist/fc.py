import torch
import torch.nn as nn
import torch.nn.functional as F
from models import base_model

class FC(base_model.HookModule):
    def __init__(self, device):
        super(FC, self).__init__(device)
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
        output = F.log_softmax(out)
        return output


def build_fc(device):
    return FC(device).to(device)
