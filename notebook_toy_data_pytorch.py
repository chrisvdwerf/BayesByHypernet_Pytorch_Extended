# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# generate toy data
from __future__ import division, print_function
import numpy as np
import pandas as pd
import seaborn as sns
import time
import warnings
import matplotlib.pyplot as plt
import os

# set gpu device - only important for multi gpu systems
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

rng = np.random.RandomState(1)

num_samples = 20
# num_samples = 10000

data_x = rng.uniform(low=-4, high=4, size=(num_samples,))
data_y = data_x ** 3 + rng.normal(loc=0, scale=9, size=(num_samples,))

linspace = np.linspace(-6, 6, num=500)

# +
# plot toy data
# %matplotlib inline

# plt.figure(figsize=(10, 7))
# plt.plot(linspace, linspace ** 3)
# plt.plot(data_x, data_y, 'ro')
# plt.show()

# dataframe to hold results
cols = ['x', 'y', 'mode', 'mc']

exps = {
    # 'bbb_pytorch': 'Bayes by Backprop (torch)',
    'vanilla': 'MAP',
    'implicit_fullkl_structured': 'Bayes by Hypernet',
    # 'mnf': 'MNF',
    'bbb': 'Bayes by Backprop',
    # 'dropout': 'MC-Dropout', 'ensemble': 'Ensemble',
    'implicit_fullnoise': 'Bayes by Hypernet with Full Noise',
    'implicit_fullkl': 'Bayes by Hypernet with Full KL',
    # 'hmc': 'Hamiltonian Monte Carlo'
}

import util.toy_data_tensorflow as tf_util
import util.toy_data_pytorch as torch_util

# from util.toy_data_tensorflow import train_and_predict


prediction_df = pd.DataFrame(columns=cols)
weight_dict = {}

for mode in exps.keys():
    if 'pytorch' in mode:  # pytorch
        dataframe, weight_dict = torch_util.train_and_predict(mode, data_x, data_y)
    else:  # tensorflow
        dataframe, weight_dict = tf_util.train_and_predict(mode, data_x, data_y)

    prediction_df = pd.concat([prediction_df, dataframe])

prediction_df['title'] = [exps[f] for f in prediction_df['mode']]
# -

len(exps)

colours = sns.color_palette(n_colors=9)

# +
# t = {'mnf': 'MNF', 'bbb': 'Bayes by Backprop', 'implicit': 'Bayes by Hypernet',
#     'dropout': 'MC-Dropout', 'vanilla': 'MAP', 'ensemble': 'Ensemble'}
t = exps

prediction_df['title'] = [exps[f] for f in prediction_df['mode']]
# -

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fig, axes = plt.subplots(1, 9, figsize=(40, 7), sharey=True)
    for i, (mode, label) in enumerate(exps.items()):
        mode_df = prediction_df[prediction_df['mode'] == mode]
        # axes[i].set_title(label)
        axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
        axes[i].plot(data_x, data_y, 'o', color='black', label='Samples')
        sns.tsplot(mode_df, time='x', value='y', condition='title', unit='mc', ci='sd', ax=axes[i])
        l = axes[i].legend(loc=0)
        l.set_title('')
    plt.ylim(-100, 100)
    sns.despine()
    plt.tight_layout()
    plt.show()

colours = sns.color_palette(n_colors=9)

# +


plt.figure(figsize=(12, 7))
plt.plot(linspace, linspace ** 3, '--', label='Real function')
plt.plot(data_x, data_y, 'o', color='black', label='Samples')
for i, (mode, label) in enumerate(t.items()):
    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    plt.plot(linspace, mean_pred, color=colours[i], label=label)

    if not mode == 'vanilla':
        plt.fill_between(linspace,
                         mean_pred + 1 * std_pred,
                         mean_pred - 1 * std_pred,
                         color=colours[i], alpha=0.3)
        plt.fill_between(linspace,
                         mean_pred + 2 * std_pred,
                         mean_pred - 2 * std_pred,
                         color=colours[i], alpha=0.2)
        plt.fill_between(linspace,
                         mean_pred + 3 * std_pred,
                         mean_pred - 3 * std_pred,
                         color=colours[i], alpha=0.1)
plt.ylim(-100, 100)
l = plt.legend(loc=0)
l.set_title('')
sns.despine()
plt.show()
# -

len(exps.keys())

fig, axes = plt.subplots(1, 9, figsize=(40, 7), sharey=True)
for i, (mode, label) in enumerate(exps.items()):
    # axes[i].set_title(label)
    axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[i].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[i].plot(linspace, mean_pred, color=colours[i], label=label)

    if not mode == 'vanilla':
        axes[i].fill_between(linspace,
                             mean_pred + 1 * std_pred,
                             mean_pred - 1 * std_pred,
                             color=colours[i], alpha=0.3)
        axes[i].fill_between(linspace,
                             mean_pred + 2 * std_pred,
                             mean_pred - 2 * std_pred,
                             color=colours[i], alpha=0.2)
        axes[i].fill_between(linspace,
                             mean_pred + 3 * std_pred,
                             mean_pred - 3 * std_pred,
                             color=colours[i], alpha=0.1)

    l = axes[i].legend(loc=0)
    l.set_title('')
plt.ylim(-100, 100)
sns.despine()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 6, figsize=(40, 7), sharey=True)
for i, (mode, label) in enumerate(t.items()):
    # axes[i].set_title(label)
    axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[i].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[i].plot(linspace, mean_pred, color=colours[0], label=label)

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
for i, (mode, label) in enumerate(t.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[i], label=label)

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
for i, (mode, label) in enumerate(exps.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[i], label=label)

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
for i, (mode, label) in enumerate(exps.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[0], label=label)

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
for i, (mode, label) in enumerate(exps_small.items()):
    row = i // 3
    col = i % 3
    # axes[row, col].set_title(label)
    axes[row, col].plot(linspace, linspace ** 3, '--', label='Real function')
    axes[row, col].plot(data_x, data_y, 'o', color='black', label='Samples')

    mode_df = prediction_df[prediction_df['mode'] == mode]
    groups = mode_df.groupby(['x'])
    mean_pred = groups.mean().values[:, 0]
    std_pred = groups.std().values[:, 0]
    axes[row, col].plot(linspace, mean_pred, color=colours[0], label=label)

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

    axes[row, col].set_title(label, size='larger')
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
    for i, (mode, label) in enumerate(exps.items()):
        axes[i].set_title(label)
        axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
        axes[i].plot(data_x, data_y, '.', color='black', label='Samples')

        mode_df = prediction_df[prediction_df['mode'] == mode]
        groups = mode_df.groupby(['x'])
        mean_pred = groups.mean().as_matrix()[:, 0]
        std_pred = groups.std().as_matrix()[:, 0]
        axes[i].plot(linspace, mean_pred, color=colours[0], label=label)

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
    for i, (mode, label) in enumerate(exps.items()):
        axes[i].set_title(label)
        axes[i].plot(linspace, linspace ** 3, '--', label='Real function')
        axes[i].plot(data_x, data_y, '.', color='black', label='Samples')

        mode_df = prediction_df[prediction_df['mode'] == mode]
        groups = mode_df.groupby(['x'])
        mean_pred = groups.mean().as_matrix()[:, 0]
        std_pred = groups.std().as_matrix()[:, 0]
        axes[i].plot(linspace, mean_pred, color=colours[0], label=label)

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
# -
