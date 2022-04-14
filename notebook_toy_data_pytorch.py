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

# FIG_NAME = "figures/hypernet_noise1_shared_tensorflow"
# FIG_NAME = "figures/hypernet_noise1_shared_pytorch"
# FIG_NAME = "figures/hypernet_noise100_tensorflow"
# FIG_NAME = "figures/hypernet_noise100_pytorch" # fullnoise
FIG_NAME = "figures/hypernet_noise200_shared_tensorflow"
# FIG_NAME = "figures/hypernet_noise200_shared_pytorch"
# kl
# kl

STD_DEV_MULT = 1
LEGEND = False
exps = {
    **generate_N("implicit_fullnoisesh", merge_dicts(basic_config, {
        "desc": 'Bayes by Hypernet (torch)',
        "training_it": 200,
        "lr": 0.01,
        "noise_shape": 200,
        "noise_sharing": True
    }), 5),


    # **generate_N("bbb", merge_dicts(basic_config, {
    #     "desc": 'Bayes by Backprop (torch)',
    #     "training_it": 40,
    #     "lr": 0.1,
    #     "loss_crit": torch.nn.MSELoss()
    # }), 5),
    # **generate_N("ensemble_pytorch", merge_dicts(basic_config, {
    #     "desc": "Ensemble",
    #     "adv_alpha": 0.5,
    #     "adv_eps": 1e-2,
    #     "training_it": 40,
    #     "lr": 0.1,
    #     "num_net": 10,
    #     "mc_steps": 42,  # note: no impact; ensembles is deterministic, computes average of N networks
    # }), 5),
    # **generate_N("dropout", merge_dicts(basic_config, {
    #     "desc": "Dropout",
    #     "training_it": 40,
    #     "lr": 0.1,
    #     "mc_steps": 42,  # note: no impact; ensembles is deterministic, computes average of N networks
    # }), 5),
    # **generate_N("implicit_pytorch", merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    #     "lr": 0.02,
    #     "noise_shape": 200,
    #     "noise_sharing": True
    # }), 8),
    # **generate_N("implicit_fullnoise", merge_dicts(basic_config, {
        # 'implicit_fullnoise': 'Bayes by Hypernet with Full Noise',
    # **generate_N("implicit_fullkl_structured", merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     # "training_it": 200,
    #     # "noise_shape": 200,
    #     # "noise_sharing": True
    # }), 8),
    # 'ensemble_2': {"desc": 'Ensemble'},
    # 'ensemble_3': {"desc": 'Ensemble'},
    # 'ensemble_4': {"desc": 'Ensemble'},
    # 'ensemble_5': {"desc": 'Ensemble'},
    # 'ensemble_6': {"desc": 'Ensemble'},
    # 'ensemble_8': {"desc": 'Ensemble'},
    # 'ensemble_9': {"desc": 'Ensemble'},
    # 'ensemble_3': 'Ensemble',
    # 'ensemble_4': 'Ensemble',
    # 'ensemble_5': 'Ensemble',
    # 'ensemble_6': 'Ensemble',
    # 'ensemble_8': 'Ensemble',

    # 'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    #     "noise_shape": 200,
    #     "noise_sharing": True
    # }),
    # 'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    # }),
    # 'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    # }),    'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    # }),    'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    # }),    'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    # }),    'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    # }),    'implicit_pytorch': merge_dicts(basic_config, {
    #     "desc": 'Bayes by Hypernet (torch)',
    #     "training_it": 200,
    # }),

    # 'ensemble_2_pytorch': 'Ensemble',
    # 'ensemble_3_pytorch': 'Ensemble',
    # 'ensemble_4_pytorch': 'Ensemble',
    # 'ensemble_5_pytorch': 'Ensemble',
    # 'ensemble_6_pytorch': 'Ensemble',
    # 'ensemble_8_pytorch': 'Ensemble',
    # 'ensemble_9_pytorch': 'Ensemble',
    #
    #
    #
    #
    # 'vanilla': 'MAP',
    # 'ensemble_1_pytorch': 'Ensemble',
    # 'bbb_pytorch': 'Bayes by Backprop (pytorch)',
    # 'implicit_pytorch_1_ns200_': 'Bayes by Hypernet (torch)',
    # 'dropout1_pytorch': 'MC-Dropout',

    # 'ensemble_2_pytorch': 'Ensemble',
    # 'ensemble_3_pytorch': 'Ensemble',
    # 'ensemble_4_pytorch': 'Ensemble',
    # 'ensemble_5_pytorch': 'Ensemble',
    # 'ensemble_6_pytorch': 'Ensemble',
    # 'ensemble_8_pytorch': 'Ensemble',

    # 'ensemble_2': 'Ensemble',
    # 'ensemble_3': 'Ensemble',
    # 'ensemble_4': 'Ensemble',
    # 'ensemble_5': 'Ensemble',
    # 'ensemble_6': 'Ensemble',
    # 'ensemble_8': 'Ensemble',
    # 'fqwfq': 'qwidhqw',
    # 'implicit_pytorch_1_ns200_': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_2_ns200_': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_3_ns200_': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_4_ns200_': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_5_ns200_': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_6_ns200_': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_8_ns200_': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_3': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_4': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_5': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_6': 'Bayes by Hypernet (torch)',
    # 'implicit_pytorch_': 'Bayes by Hypernet (torch)',
    # 'implicit_fullkl_structured1': 'Bayes by Hypernet',
    # 'implicit_fullkl_structured2': 'Bayes by Hypernet',
    # 'implicit_fullkl_structured3': 'Bayes by Hypernet',
    # 'implicit_fullkl_structured4': 'Bayes by Hypernet',
    # 'implicit_fullkl_structured5': 'Bayes by Hypernet',
    # 'implicit_fullkl_structured6': 'Bayes by Hypernet',
    # 'implicit_fullkl_structured8': 'Bayes by Hypernet',

    # 'implicit_fullkl_structured': 'Bayes by Hypernet',
    # 'mnf': 'MNF',
    # 'bbb': 'Bayes by Backprop',
    # 'dropout1_pytorch': 'MC-Dropout',
    # 'dropout2_pytorch': 'MC-Dropout',
    # 'dropout3_pytorch': 'MC-Dropout',
    # 'dropout4_pytorch': 'MC-Dropout',
    # 'dropout5_pytorch': 'MC-Dropout',
    # 'dropout6_pytorch': 'MC-Dropout',
    # 'dropout8_pytorch': 'MC-Dropout',

    # 'ensemble': 'Ensemble',
    # 'implicit_fullnoise': 'Bayes by Hypernet with Full Noise',
    # 'implicit_fullkl': 'Bayes by Hypernet with Full KL',
    # 'hmc': 'Hamiltonian Monte Carlo'
    # 'bbb_pytorch': 'Bayes by Backprop (pytorch)',
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

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     fig, axes = plt.subplots(1, 9, figsize=(40, 7), sharey=True)
#     for i, (mode, label) in enumerate(exps.items()):
#         mode_df = prediction_df[prediction_df['mode'] == mode]
#         # axes[i].set_title(label)
#         axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
#         axes[i].plot(data_x, data_y, 'o', color='black', label='Samples')
#         sns.tsplot(mode_df, time='x', value='y', condition='title', unit='mc', ci='sd', ax=axes[i])
#         l = axes[i].legend(loc=0)
#         l.set_title('')
#     plt.ylim(-100, 100)
#     sns.despine()
#     plt.tight_layout()
#     plt.show()

colours = sns.color_palette(n_colors=9)

# +

#
# # plt.figure(figsize=(12, 7))
# plt.figure(figsize=(12, len(t.items())))
#
# plt.plot(linspace, linspace ** 3, '--', label='Real function')
# plt.plot(data_x, data_y, 'o', color='black', label='Samples')
# for i, (mode, config) in enumerate(t.items()):
#     mode_df = prediction_df[prediction_df['mode'] == mode]
#     groups = mode_df.groupby(['x'])
#     mean_pred = groups.mean().values[:, 0]
#     std_pred = groups.std().values[:, 0]
#     plt.plot(linspace, mean_pred, color=colours[i], label=config['desc'])
#
#     if not mode == 'vanilla':
#         plt.fill_between(linspace,
#                          mean_pred + 1 * std_pred,
#                          mean_pred - 1 * std_pred,
#                          color=colours[i], alpha=0.3)
#         plt.fill_between(linspace,
#                          mean_pred + 2 * std_pred,
#                          mean_pred - 2 * std_pred,
#                          color=colours[i], alpha=0.2)
#         plt.fill_between(linspace,
#                          mean_pred + 3 * std_pred,
#                          mean_pred - 3 * std_pred,
#                          color=colours[i], alpha=0.1)
# plt.ylim(-100, 100)
# l = plt.legend(loc=0)
# l.set_title('')
# sns.despine()
# plt.show()
# # -

len(exps.keys())


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
        l.set_title('')
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_NAME)
# plt.show()

exit(0)
fig, axes = plt.subplots(1, 6, figsize=(40, 7), sharey=True)
for i, (mode, config) in enumerate(t.items()):
    # axes[i].set_title(label)
    axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[i].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[i].plot(linspace, mean_pred, color=colours[0], label=config['desc'])

    if not mode == 'vanilla':
        axes[i].fill_between(linspace,
                             mean_pred + 1 * std_pred,
                             mean_pred - 1 * std_pred,
                             color=colours[0], alpha=0.3)
        axes[i].fill_between(linspace,
                             mean_pred + 2 * std_pred,
                             mean_pred - 2 * std_pred,
                             color=colours[0], alpha=0.2)
        axes[i].fill_between(linspace,
                             mean_pred + 3 * std_pred,
                             mean_pred - 3 * std_pred,
                             color=colours[0], alpha=0.1)

    l = axes[i].legend(loc=0)
    l.set_title('')
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(20, 14), sharey=True)
for i, (mode, config) in enumerate(t.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[i], label=config)

    if not mode == 'vanilla':
        axes[row, col].fill_between(linspace,
                                    mean_pred + 1 * std_pred,
                                    mean_pred - 1 * std_pred,
                                    color=colours[i], alpha=0.3)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 2 * std_pred,
                                    mean_pred - 2 * std_pred,
                                    color=colours[i], alpha=0.2)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 3 * std_pred,
                                    mean_pred - 3 * std_pred,
                                    color=colours[i], alpha=0.1)

    l = axes[row, col].legend(loc=0)
    l.set_title('')
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharey=True)
for i, (mode, config) in enumerate(exps.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[i], label=config)

    if not mode == 'vanilla':
        axes[row, col].fill_between(linspace,
                                    mean_pred + 1 * std_pred,
                                    mean_pred - 1 * std_pred,
                                    color=colours[i], alpha=0.3)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 2 * std_pred,
                                    mean_pred - 2 * std_pred,
                                    color=colours[i], alpha=0.2)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 3 * std_pred,
                                    mean_pred - 3 * std_pred,
                                    color=colours[i], alpha=0.1)

    l = axes[row, col].legend(loc=0)
    l.set_title('')
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(17, 10), sharey=True)
for i, (mode, config) in enumerate(exps.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[0], label=config)

    if not mode == 'vanilla':
        axes[row, col].fill_between(linspace,
                                    mean_pred + 1 * std_pred,
                                    mean_pred - 1 * std_pred,
                                    color=colours[0], alpha=0.3)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 2 * std_pred,
                                    mean_pred - 2 * std_pred,
                                    color=colours[0], alpha=0.2)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 3 * std_pred,
                                    mean_pred - 3 * std_pred,
                                    color=colours[0], alpha=0.1)

    l = axes[row, col].legend(loc=0)
    l.set_title('')
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.show()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for key in weight_dict.keys():
        if key == 'MAP' or key == 'vanilla':
            continue
        print(exps[key])
        weight_dist = np.squeeze(np.array(weight_dict[key]))

        fig, axes = plt.subplots(20, 10, figsize=(40, 40), sharey=False)
        for i in range(200):
            row = i // 10
            col = i % 10

            sns.kdeplot(weight_dist[:, i], ax=axes[row, col], shade=True)
        plt.tight_layout()
        sns.despine()
        plt.show()

# +
### more data - 3k HMC samples
# -


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for key in weight_dict.keys():
        print(exps[key])
        if key == 'MAP' or key == 'vanilla':
            continue
        weight_dist = np.squeeze(np.array(weight_dict[key]))

        fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharey=False)
        for i in range(10):
            row = i // 5
            col = i % 5

            sns.kdeplot(weight_dist[:, i + row * 100].astype(np.float32), ax=axes[row, col], shade=True)
        sns.despine()
        plt.tight_layout()
        plt.show()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for key in weight_dict.keys():
        print(exps[key])
        if key == 'MAP' or key == 'vanilla':
            continue
        weight_dist = np.squeeze(np.array(weight_dict[key]))

        fig, ax = plt.subplots(2, 5, figsize=(10, 3))
        for i in range(10):
            row = i // 5
            col = i % 5

            sns.kdeplot(weight_dist[:, i + row * 100].astype(np.float32), ax=ax[row, col], shade=True)
        sns.despine()
        plt.tight_layout()
        plt.show()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for key in weight_dict.keys():
        print(key)
        if key == 'MAP' or key == 'vanilla':
            continue
        weight_dist = np.squeeze(np.array(weight_dict[key]))

        w_df = pd.DataFrame(data=weight_dist)
        corr_df = w_df.corr()
        print(corr_df.shape)
        plt.figure(figsize=(15, 15))
        sns.heatmap(corr_df, vmin=-1, vmax=1.)
        plt.xticks([])
        plt.yticks([])
        plt.show()

w_df.shape

# +
n = tf.shape(gen_weights)[0]
mu = tf.ones((n,))

f1 = tf.concat([tf.ones((tf.cast((n * (n + 1)) / 2 - 100, tf.int32),)), tf.zeros((100,))], 0)
scale_tril1 = tf.eye(n) + tfd.fill_triangular(f1)
d1 = tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril1)

f2 = tf.concat([tf.zeros((tf.cast((n * (n + 1)) / 2 - 100, tf.int32),)), tf.ones((100,))], 0)
scale_tril2 = tf.eye(n) + tfd.fill_triangular(f2)
d2 = tfd.MultivariateNormalTriL(loc=-1 * mu, scale_tril=scale_tril2)

tfd = tfp.distributions
mix = 0.5
bimix_gauss = tfd.Mixture(
    cat=tfd.Categorical(probs=[mix, 1. - mix]),
    components=[
        d1, d2
    ])
prior_samples = tf.transpose(bimix_gauss.sample(num_samples), [1, 0])
# -

mapping = {0: 'w1', 1: 'b1', 2: 'w2', 3: 'b2'}
for key in weight_dict.keys():
    print(mapping[key])
    weight_dist = np.squeeze(np.array(weight_dict[key]))
    if len(weight_dist.shape) == 1:
        plt.figure(figsize=(7, 7))
        sns.distplot(weight_dist)
        plt.show()
    else:
        fig, axes = plt.subplots(10, 10, figsize=(40, 40), sharey=False)
        for i in range(100):
            row = i // 10
            col = i % 10

            sns.distplot(weight_dist[:, i], ax=axes[row, col])
        plt.tight_layout()
        plt.show()

# +
prediction_df.to_csv('toy_example.csv')

try:
    import cPickle as pickle
except:
    import pickle

with open('toy_bbh_weight.pickle', 'wb') as f:
    pickle.dump(weight_dict, f)
# -

plt.figure(figsize=(7, 7))
sns.distplot(weight_dict[2][-3], bins=20)
sns.despine()
plt.xlabel('Weight value $w$')
plt.ylabel('$p(w)$')
plt.show()

# +
### plot for paper

# +
exps_small = {'mnf': 'MNF',
              'bbb': 'Bayes by Backprop',
              'implicit_fullkl': 'Bayes by Hypernet',
              'dropout': 'MC-Dropout',
              'vanilla': 'MAP',
              'ensemble': 'Ensemble'}

fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharey=True)
for i, (mode, config) in enumerate(exps_small.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[0], label=config)

    if not mode == 'vanilla':
        axes[row, col].fill_between(linspace,
                                    mean_pred + 1 * std_pred,
                                    mean_pred - 1 * std_pred,
                                    color=colours[0], alpha=0.3)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 2 * std_pred,
                                    mean_pred - 2 * std_pred,
                                    color=colours[0], alpha=0.2)
        axes[row, col].fill_between(linspace,
                                    mean_pred + 3 * std_pred,
                                    mean_pred - 3 * std_pred,
                                    color=colours[0], alpha=0.1)

    axes[row, col].set_title(config, size='larger')
    # l = axes[row, col].legend(loc=0)
    # l.set_title('')
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.show()

# +
exps = {
    'vanilla': 'MAP', 'implicit_fullkl_structured': 'BbH struc. prior',
    'mnf': 'MNF',
    'bbb': 'Bayes by Backprop',
    'dropout': 'MC-Dropout', 'ensemble': 'Ensemble',
    'implicit_fullnoise': 'BbH full noise',
    'implicit_fullkl': 'BbH with Full KL',
    'hmc': 'HMC'
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fig, axes = plt.subplots(1, 9, figsize=(22, 2.7), sharey=True)
    for i, (mode, config) in enumerate(exps.items()):
        axes[i].set_title(config)
        axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
        axes[i].plot(data_x, data_y, '.', color='black', label='Samples')

        mode_df = prediction_df[prediction_df['mode'] == mode]
        groups = mode_df.groupby(['x'])
        mean_pred = groups.mean().as_matrix()[:, 0]
        std_pred = groups.std().as_matrix()[:, 0]
        axes[i].plot(linspace, mean_pred, color=colours[0], label=config)

        if not mode == 'vanilla':
            axes[i].fill_between(linspace,
                                 mean_pred + 1 * std_pred,
                                 mean_pred - 1 * std_pred,
                                 color=colours[0], alpha=0.3)
            axes[i].fill_between(linspace,
                                 mean_pred + 2 * std_pred,
                                 mean_pred - 2 * std_pred,
                                 color=colours[0], alpha=0.2)
            axes[i].fill_between(linspace,
                                 mean_pred + 3 * std_pred,
                                 mean_pred - 3 * std_pred,
                                 color=colours[0], alpha=0.1)

    # l = axes[i].legend(loc=0)
    # l.set_title(label)
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.show()

# +
exps = {
    'vanilla': 'MAP',
    # 'implicit_fullkl_structured': 'Bayes by Hypernet',
    'mnf': 'MNF',
    'bbb': 'Bayes by Backprop',
    'dropout': 'MC-Dropout', 'ensemble': 'Ensemble',
    # 'implicit_fullnoise': 'Bayes by Hypernet with Full Noise',
    'implicit_fullkl': 'Bayes by Hypernet',
    'hmc': 'HMC'
}

fig, axes = plt.subplots(1, 7, figsize=(15, 2.7), sharey=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i, (mode, config) in enumerate(exps.items()):
        axes[i].set_title(config)
        axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
        axes[i].plot(data_x, data_y, '.', color='black', label='Samples')

        mode_df = prediction_df[prediction_df['mode'] == mode]
        groups = mode_df.groupby(['x'])
        mean_pred = groups.mean().as_matrix()[:, 0]
        std_pred = groups.std().as_matrix()[:, 0]
        axes[i].plot(linspace, mean_pred, color=colours[0], label=config)

        if not mode == 'vanilla':
            axes[i].fill_between(linspace,
                                 mean_pred + 1 * std_pred,
                                 mean_pred - 1 * std_pred,
                                 color=colours[0], alpha=0.3)
            axes[i].fill_between(linspace,
                                 mean_pred + 2 * std_pred,
                                 mean_pred - 2 * std_pred,
                                 color=colours[0], alpha=0.2)
            axes[i].fill_between(linspace,
                                 mean_pred + 3 * std_pred,
                                 mean_pred - 3 * std_pred,
                                 color=colours[0], alpha=0.1)

        # l = axes[i].legend(loc=0)
        # l.set_title(label)
    plt.ylim(-100, 100)
    sns.despine()
    plt.tight_layout()
    plt.show()
