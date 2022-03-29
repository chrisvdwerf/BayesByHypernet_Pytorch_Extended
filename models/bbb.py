import torch.nn as nn
import torch as th
import numpy as np

from typing import List


def log_gauss(mean: th.Tensor, std: th.Tensor, x: th.Tensor) -> th.Tensor:
    return - ((x - mean) ** 2) / (2 * std ** 2) \
           + -th.log(th.abs(std)) - 0.5 * np.log(2 * np.pi)


def std_from_rho(rho: th.Tensor) -> th.Tensor:
    return th.log(1 + th.exp(rho))


class BBBLayer(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_weight_samples: int, init_mean: float = 0.,
                 init_std: float = 1.):
        super().__init__()
        self.n_weight_samples = n_weight_samples
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.mu_w = nn.Parameter(th.randn((n_inputs, n_outputs)) * init_std + init_mean,
                                 requires_grad=True)
        self.rho_w = nn.Parameter(th.randn((n_inputs, n_outputs)) * init_std + init_mean,
                                  requires_grad=True)

        self.mu_b = nn.Parameter(th.randn(n_outputs) * init_std + init_mean, requires_grad=True)
        self.rho_b = nn.Parameter(th.randn(n_outputs) * init_std + init_mean, requires_grad=True)

        # these fields temporarily store all weight samples after a forward pass
        self.sampled_w = th.empty(n_weight_samples, n_inputs, n_outputs, requires_grad=False)
        self.sampled_b = th.empty(n_weight_samples, n_outputs, requires_grad=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[0] == self.n_weight_samples
        # x.shape[1] is the number of data samples
        assert x.shape[2] == self.n_inputs

        y = th.empty((self.n_weight_samples, x.shape[1], self.n_outputs))

        for i in range(self.n_weight_samples):
            epsilon_w = th.randn(self.rho_w.shape)  # could also have picked mu_w -> same shape
            epsilon_b = th.randn(self.rho_b.shape)

            # sample new weights and biases using the random epsilon
            w = self.mu_w + std_from_rho(self.rho_w) * epsilon_w
            b = self.mu_b + std_from_rho(self.rho_b) * epsilon_b

            # store the sampled weights and biases
            self.sampled_w[i] = w
            self.sampled_b[i] = b

            # calculate output based on sampled weights
            y[i] = th.mm(x[i], w) + b

        return y


# (n_weight_samples, n_data_samples, n_features)
# x: (n_weight_samples, n_data_samples, n_inputs) <- if not the first layer, x might have different
#       values over the weight sample dimension.


def bbb_criterion(y_preds: th.Tensor, y_target: th.Tensor, bbb_layers: List[BBBLayer],
                  simplicity_prior: float = 1.) -> th.Tensor:
    # check the shapes: y: (n_weight_samples, n_data_samples, n_outputs)
    assert y_preds.shape[1] == y_target.shape[0]
    assert y_preds.shape[2] == y_target.shape[1]
    n_weight_samples = y_preds.shape[0]

    # all layers should have the same n_weight_samples value
    assert all([n_weight_samples == layer.n_weight_samples for layer in bbb_layers])

    # calculate the model likelihood term and the model prior term of the loss (derived from KL)
    model_likelihood_loss = 0
    model_prior_loss = 0

    for i in range(n_weight_samples):
        for layer in bbb_layers:

            # add model likelihood loss for this particular layer of this weight sample
            w_likelihood = th.sum(log_gauss(layer.mu_w, std_from_rho(layer.rho_w),
                                            layer.sampled_w[i]))
            b_likelihood = th.sum(log_gauss(layer.mu_b, std_from_rho(layer.rho_b),
                                            layer.sampled_b[i]))

            model_likelihood_loss += w_likelihood + b_likelihood

            # add model prior loss for this particular layer of this weight sample
            w_prior = th.sum(log_gauss(th.zeros_like(layer.sampled_w[0]),
                                       th.full_like(layer.sampled_w[0], simplicity_prior),
                                       layer.sampled_w[i]))
            b_prior = th.sum(log_gauss(th.zeros_like(layer.sampled_b[0]),
                                       th.full_like(layer.sampled_b[0], simplicity_prior),
                                       layer.sampled_b[i]))

            model_prior_loss -= w_prior + b_prior

    # calculate the data dependent term of the loss (prediction loss)
    prediction_loss = 0

    for i in range(n_weight_samples):
        prediction_loss += 0.5 * th.nn.MSELoss()(y_preds[i], y_target)

    total_loss = model_likelihood_loss + model_prior_loss + prediction_loss
    return total_loss / n_weight_samples  # normalize by number of weight samples


# y: (n_weight_samples, n_data_samples, n_outputs)
# y_preds: (n_data_samples, n_outputs)

# assumptions:
# 1. resample epsilon for every weight
# 2. derivation of the loss
