import pandas as pd
import numpy as np
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import trange
from util.models.dropout import DropoutNN
from .models.bbh import ToyNN
import torch.distributions as dist

hidden = 100

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def train_and_predict(mode: str, data_x, data_y, linspace=np.linspace(-6, 6, num=500), seed_int=42, name=None) -> (
pd.DataFrame, dict):
    """
    mode - string of the mode
    data_x - datapoints x for training
    data_y - datapoints y for training
    linspace - prediction samples x
    --- output ---
    dataframe - all predictions in linspace
    weight_dict - dictionary with one key for the weights of the model used defined by mode
    """
    cols = ['x', 'y', 'mode', 'mc']
    prediction_df = pd.DataFrame(columns=cols)
    weight_dict = {}
    seed(seed_int)
    name = name if name is not None else mode # name in dataframe

    batch_x = torch.from_numpy(data_x.astype(np.float32).reshape(20, 1))
    batch_y = torch.from_numpy(data_y.astype(np.float32).reshape(20, 1))

    crit = lambda x, y: torch.sum(-1 * dist.Normal(0., 9.).log_prob(x - y))

    if "implicit" in mode:
        model = ToyNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-5)

        # === TRAIN ===
        with trange(200) as pbar:
            for i in pbar:
                optimizer.zero_grad()
                cur_anneal = np.clip(10. / (i + 1) - 1., 0., 1.)
                preds = model(batch_x)
                mse = crit(preds, batch_y)
                kl = model.kl()

                loss = mse + cur_anneal * kl

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.detach().numpy(), mse=mse.detach().numpy(), kl=kl.detach().numpy())
            # pbar.close() niet nodig??

        # === PREDICT ===
        prediction_df = pd.DataFrame(columns=cols)
        mcsteps = 100
        all_preds = np.zeros(len(linspace))
        batch_x = torch.from_numpy(linspace[:, np.newaxis].astype(np.float32))
        for mc in range(mcsteps):
            with torch.no_grad():
                predictions = model(batch_x)[:, 0]
            all_preds += predictions.numpy() / mcsteps
            new_df = pd.DataFrame(columns=cols, data=list(zip(
                linspace, predictions.numpy(), [name] * len(linspace), [mc] * len(linspace))))

            prediction_df = pd.concat([prediction_df, new_df])
    elif mode == 'dropout_torch':
        model = DropoutNN(batch_x)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.1, eps=1e-5)
        with trange(40) as pbar:
            for i in pbar:
                optimiser.zero_grad()
                preds = model(batch_x)
                loss = torch.mean(crit(preds, batch_y))
                optimiser.step()
                pbar.set_postfix(loss=loss)
            pbar.close()
        
        prediction_df = pd.DataFrame(columns=cols)
        mcsteps = 100
        all_preds = np.zeros(len(linspace))
        torch.from_numpy(linspace[:, np.newaxis].astype(np.float32))
        for mc in range(mcsteps):
            with torch.no_grad():
                predictions = model(batch_x)[:, 0]
            all_preds += predictions.numpy() / mcsteps
            new_df = pd.DataFrame(columns=cols, data=list(zip(
                linspace, predictions.numpy(), [mode] * len(linspace), [mc] * len(linspace))))

            prediction_df = pd.concat([prediction_df, new_df])
    else:
        raise NotImplementedError("")

    return prediction_df, weight_dict