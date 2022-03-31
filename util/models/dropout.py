import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as dist

class DropoutNN(nn.module):
    def __init__(self, units=[16, 32, 64]):
        super(DropoutNN, self).__init__()
        self.layer1_w = torch.zeros((100, 1))
        self.layer1_b = torch.zeros((100, ))
        self.layer2_w = torch.zeros((100, 1))
        self.layer2_b = torch.zeros((1, ))
    
    def forward(self, x):
        w1 = self.layer1_w
        b1 = self.layer1_b

        w2 = self.layer2_w
        b2 = self.layer2_b

        x = F.linear(x, w1, b1)
        x = F.relu(x)
        x = F.dropout(0.5)
        x = F.linear(x, w2, b2)