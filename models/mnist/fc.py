import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self,x ):
        x = torch.flatten(x, 1)
        x = self.fc1(x);
        x = F.relu(x, inplace=True)

        x = self.fc2(x)
        x = F.relu(x, inplace=True)

        x = self.fc3(x)

        output = F.log_softmax(x)
        return output


def build_fc():
    return FC()
