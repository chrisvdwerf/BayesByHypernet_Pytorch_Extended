import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as dist


class HypernetWeight(nn.Module):
    def __init__(self, shape, units=[16, 32, 64], bias=True,
                 noise_shape=1, activation=nn.LeakyReLU(0.1)):
        super(HypernetWeight, self).__init__()
        self.shape = shape
        self.noise_shape = noise_shape

        layers = []
        in_features = noise_shape
        for out_features in units:
            layers.append(nn.Linear(in_features, out_features, bias=bias))
            layers.append(activation)
            in_features = out_features

        layers.append(nn.Linear(in_features, np.prod(shape), bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x=None, num_samples=1):
        if x is None:
            x = torch.randn((num_samples, self.noise_shape))
        return self.net(x).reshape((x.shape[0], *self.shape))


class ToyNN(nn.Module):
    def __init__(self, units=[16, 32, 64], noise_shape=1, noise_sharing=True):
        super(ToyNN, self).__init__()
        self.noise_shape = noise_shape
        self.noise_sharing = noise_sharing
        self.layer1_w = HypernetWeight((100, 1), units=units, noise_shape=noise_shape)
        self.layer1_b = HypernetWeight((100,), units=units, noise_shape=noise_shape)
        self.layer2_w = HypernetWeight((1, 100), units=units, noise_shape=noise_shape)
        self.layer2_b = HypernetWeight((1,), units=units, noise_shape=noise_shape)

    def _generate_noise(self):
        return torch.randn((1, self.noise_shape))

    def forward(self, x):
        nw1 = self._generate_noise()
        if self.noise_sharing:
            nb1, nw2, nb2 = nw1, nw1, nw1
        else:
            nb1, nw2, nb2 = self._generate_noise(), self._generate_noise(), self._generate_noise()

        w1 = self.layer1_w(nw1)[0]
        b1 = self.layer1_b(nb1)[0]

        w2 = self.layer2_w(nw2)[0]
        b2 = self.layer2_b(nb2)[0]

        x = F.linear(x, w1, b1)
        x = F.relu(x)
        x = F.linear(x, w2, b2)

        return x

    def sample(self, num_samples=5):
        l1_w_samples = self.layer1_w(num_samples=num_samples).view((num_samples, -1))
        l1_b_samples = self.layer1_b(num_samples=num_samples).view((num_samples, -1))
        l2_w_samples = self.layer2_w(num_samples=num_samples).view((num_samples, -1))
        l2_b_samples = self.layer2_b(num_samples=num_samples).view((num_samples, -1))

        gen_weights = torch.cat([l1_w_samples, l1_b_samples, l2_w_samples, l2_b_samples], 1)

        return gen_weights

    def kl(self, num_samples=5, full_kernel=True):

        gen_weights = self.sample(num_samples=num_samples)
        gen_weights = gen_weights.transpose(1, 0)
        prior_samples = torch.randn_like(gen_weights)

        eye = torch.eye(num_samples, device=gen_weights.device)
        wp_distances = (prior_samples.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2
        # [weights, samples, samples]

        ww_distances = (gen_weights.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2

        if full_kernel:
            wp_distances = torch.sqrt(torch.sum(wp_distances, 0) + 1e-8)
            wp_dist = torch.min(wp_distances, 0)[0]

            ww_distances = torch.sqrt(
                torch.sum(ww_distances, 0) + 1e-8) + eye * 1e10
            ww_dist = torch.min(ww_distances, 0)[0]

            # mean over samples
            kl = torch.mean(torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8))
            kl *= gen_weights.shape[0]
            kl += np.log(float(num_samples) / (num_samples - 1))
        else:
            wp_distances = torch.sqrt(wp_distances + 1e-8)
            wp_dist = torch.min(wp_distances, 1)[0]

            ww_distances = (torch.sqrt(ww_distances + 1e-8)
                            + (eye.unsqueeze(0) * 1e10))
            ww_dist = torch.min(ww_distances, 1)[0]

            # sum over weights, mean over samples
            # Basically calculating the divergence for each
            # weight, instead of for the entire sample like with full_kernels
            kl = torch.sum(torch.mean(
                torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8)
                + torch.log(float(num_samples) / (num_samples - 1)), 1))

        return kl
