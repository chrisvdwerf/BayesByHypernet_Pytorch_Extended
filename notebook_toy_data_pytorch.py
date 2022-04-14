from __future__ import division, print_function
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import time
import warnings
import matplotlib.pyplot as plt
import os
import torch.distributions as dist

# set gpu device - only important for multi gpu systems
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

rng = np.random.RandomState(1)

num_samples = 20

data_x = rng.uniform(low=-4, high=4, size=(num_samples,))
data_y = data_x ** 3 + rng.normal(loc=0, scale=9, size=(num_samples,))

linspace = np.linspace(-6, 6, num=500)

# dataframe to hold results
cols = ['x', 'y', 'mode', 'mc']


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def generate_N(name, d, N):
    l = {}
    for i in range(N):
        l[(name + "_" + str(i))] = d
        # l.append()
    return l


basic_config = {
    "lr": 0.01,  # Optimiser, Learning Rate
    "adam_eps": 1e-5,  # 1e-5
    "training_it": 40,
    "mc_steps": 100,
    "hidden_size": 100,
    "loss_crit": lambda x, y: torch.sum(-1 * dist.Normal(0., 9.).log_prob(x - y)),
}

# FIG_NAME = "figures/hypernet_noise1_shared_pytorch"
# FIG_NAME = "figures/hypernet_noise1_shared_pytorch"
# FIG_NAME = "figures/hypernet_noise100_tensorflow"
# FIG_NAME = "figures/hypernet_noise100_pytorch" # fullnoise
FIG_NAME = "figures/hypernet_noise200_shared_tensorflow"
# FIG_NAME = "figures/hypernet_noise200_shared_pytorch"
# kl
# kl

FIG_NAME = "figures/pytorch_total"

STD_DEV_MULT = 1
LEGEND = False
exps = {
    **generate_N("map", merge_dicts(basic_config, { # dit moet veranderd worden naar de pytorch implementations
        "desc": 'MAP',
        "title": "MAP",
        "training_it": 200,
        "lr": 0.01,
        "noise_shape": 200,
        "noise_sharing": True
    }), 1),
    **generate_N("bbb_pytorch", merge_dicts(basic_config, {
        "desc": 'Bayes by Backprop',
        "title": "Bayes by Backprop",
        "training_it": 200,
        "lr": 0.01,
        "loss_crit": lambda x, y: torch.sum(-1 * dist.Normal(0., 9.).log_prob(x - y)), # ditMSE?
    }), 1),
    **generate_N("implicit_pytorch_n1", merge_dicts(basic_config, {
        "desc": 'Bayes by Hypernet (noise-1)',
        "title": "Bayes by Hypernet (noise-1)",
        "training_it": 40,
        "lr": 0.02,
        "noise_shape": 1,
        "noise_sharing": True # if noise is shared between hypernets to generate weights
    }), 1),
    **generate_N("implicit_pytorch_n200", merge_dicts(basic_config, {
        "desc": 'Bayes by Hypernet (noise-200)',
        "title": "Bayes by Hypernet (noise-200)",
        "training_it": 200, # more iterations
        "lr": 0.01, # smaller learning rate
        "noise_shape": 200,
        "noise_sharing": True
    }), 1),
    **generate_N("dropout_pytorch", merge_dicts(basic_config, {
        "desc": 'Dropout',
        "title": "Dropout",
        "training_it": 40,
        "lr": 0.01,
    }), 1),
    **generate_N("ensemble_pytorch", merge_dicts(basic_config, {
        "desc": 'Ensemble',
        "title": "Ensemble",
        "training_it": 40,
        "lr": 0.1,
        "adv_alpha": 0.5,
        "adv_eps": 1e-2,
        "num_net": 10
    }), 1),
}

import util.toy_data_tensorflow as tf_util
import util.toy_data_pytorch as torch_util

prediction_df = pd.DataFrame(columns=cols)
weight_dict = {}

for seed, mode in enumerate(exps.keys()):
    if 'pytorch' in mode:  # pytorch
        dataframe, weight_dict = torch_util.train_and_predict(mode, data_x, data_y, exps[mode], seed_int=seed)
    else:  # tensorflow
        dataframe, weight_dict = tf_util.train_and_predict(mode, data_x, data_y, seed_int=42+seed)

    prediction_df = pd.concat([prediction_df, dataframe])

prediction_df['title'] = [exps[f]['desc'] for f in prediction_df['mode']]
# -

len(exps)

colours = sns.color_palette(n_colors=9)

# +
# t = {'mnf': 'MNF', 'bbb': 'Bayes by Backprop', 'implicit': 'Bayes by Hypernet',
#     'dropout': 'MC-Dropout', 'vanilla': 'MAP', 'ensemble': 'Ensemble'}
t = exps

# prediction_df['title'] = [exps[f] for f in prediction_df['mode']]
# -

colours = sns.color_palette(n_colors=9)
plt.rcParams.update({'font.size': 24})
fig, axes = plt.subplots(1, len(exps.items()), figsize=(30, 8), sharey=True)
for i, (mode, config) in enumerate(exps.items()):
    # axes[i].set_title(label)
    axes[i].plot(linspace, linspace ** 3, '--', label='Real function', linewidth=4)
    axes[i].plot(data_x, data_y, 'o', color='black', label='Samples',  ms=10)

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[i].plot(linspace, mean_pred, color=colours[i], label=config['desc'], linewidth=4)

    if not mode == 'vanilla':
        axes[i].fill_between(linspace,
                             mean_pred + 1 * std_pred * STD_DEV_MULT,
                             mean_pred - 1 * std_pred * STD_DEV_MULT,
                             color=colours[i], alpha=0.3)
        axes[i].fill_between(linspace,
                             mean_pred + 2 * std_pred * STD_DEV_MULT,
                             mean_pred - 2 * std_pred * STD_DEV_MULT,
                             color=colours[i], alpha=0.2)
        axes[i].fill_between(linspace,
                             mean_pred + 3 * std_pred * STD_DEV_MULT,
                             mean_pred - 3 * std_pred * STD_DEV_MULT,
                             color=colours[i], alpha=0.1)

    if LEGEND:
        l = axes[i].legend(loc=0)
    plt.rcParams.update({'font.size': 15})
    axes[i].set_title('' if "title" not in config else config['title'])
    plt.rcParams.update({'font.size': 24})
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_NAME)
# plt.show()