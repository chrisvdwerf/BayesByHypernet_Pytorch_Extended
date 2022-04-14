import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as dist
from torchensemble import VotingClassifier


class Ensemble():
    def __init__(self, n: int, hidden: int, lr: float, eps: float):
        self.networks = [EnsembleAtom(hidden, lr, eps) for _ in range(n)]

    def train_batch(self, batch_x, batch_y):
        for net in self.networks:
            net.train_batch(batch_x, batch_y)

    def predict(self, batch_x):
        return np.array([net(batch_x) for net in self.networks])


class EnsembleAtom(nn.Module):
    def __init__(self, hidden: int, lr: float, eps: float):
        super(EnsembleAtom, self).__init__()
        self.layer1 = nn.Linear(1, hidden)
        self.layer2 = nn.Linear(hidden , 1)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr, eps=eps)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

    def train_batch(self, batch_x, batch_y):
        self.optimiser.zero_grad()
        preds = self(batch_x)
        loss = torch.mean(self.crit(preds, batch_y))

        loss.backward()
        self.optimiser.step()
        # pbar.set_postfix(loss=loss.detach().numpy())
