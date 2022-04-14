import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as dist

class DropoutNN(nn.Module):
    def __init__(self, hidden: int):
        super(DropoutNN, self).__init__()
        self.layer1 = nn.Linear(1, hidden)
        self.layer2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.layer2(x)

        return x
