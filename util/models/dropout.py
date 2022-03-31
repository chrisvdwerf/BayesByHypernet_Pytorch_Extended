import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as dist


class DropoutNN(nn.Module):
    def __init__(self):
        super(DropoutNN, self).__init__()
        self.layer1_w = nn.Linear(1, 100, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        self.layer2_w = nn.Linear(100, 1, bias=True)#torch.zeros((100, 1))


    def forward(self, x):
        x = self.layer1_w(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2_w(x)

        return x
