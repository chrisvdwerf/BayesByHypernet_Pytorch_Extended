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
from util.models.bbb import BBBLayer, bbb_criterion
from util.models.bbh import ToyNN
import torch.distributions as dist

from util.models.ensemble import Ensemble

# import torchensemble.
hidden = 100


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def train_and_predict(mode: str, data_x, data_y, config={}, linspace=np.linspace(-6, 6, num=500), seed_int=42,
                      name=None) -> (
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
    name = name if name is not None else mode  # name in dataframe

    batch_x = torch.from_numpy(data_x.astype(np.float32).reshape(20, 1))
    batch_y = torch.from_numpy(data_y.astype(np.float32).reshape(20, 1))

    CONFIG_MC_STEPS = config['mc_steps']
    CONFIG_LR = config['lr']
    CONFIG_EPS = config['adam_eps']
    CONFIG_TRAINING_IT = config['training_it']
    CONFIG_LOSS_CRIT = config['loss_crit']
    CONFIG_HIDDEN_SIZE = config['hidden_size']

    assert None not in [CONFIG_MC_STEPS, CONFIG_LR, CONFIG_EPS, CONFIG_TRAINING_IT, CONFIG_LOSS_CRIT,
                        CONFIG_HIDDEN_SIZE]

    model = None
    if "implicit" in mode:
        CONFIG_NOISE_SHAPE = config['noise_shape']
        CONFIG_NOISE_SHARING = config['noise_sharing']

        assert None not in [CONFIG_NOISE_SHAPE, CONFIG_NOISE_SHARING]

        model = ToyNN(hidden=CONFIG_HIDDEN_SIZE, noise_shape=CONFIG_NOISE_SHAPE, noise_sharing=CONFIG_NOISE_SHARING)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG_LR, eps=CONFIG_EPS)

        # === TRAIN ===
        with trange(CONFIG_TRAINING_IT) as pbar:
            for i in pbar:
                optimizer.zero_grad()
                cur_anneal = np.clip(10. / (i + 1) - 1., 0., 1.)
                preds = model(batch_x)
                error = CONFIG_LOSS_CRIT(preds, batch_y)
                kl = model.kl()

                loss = error + cur_anneal * kl

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.detach().numpy(), mse=error.detach().numpy(), kl=kl.detach().numpy())
            # pbar.close() niet nodig??

        # === PREDICT ===
        prediction_df = pd.DataFrame(columns=cols)
        mcsteps = CONFIG_MC_STEPS
        all_preds = np.zeros(len(linspace))
        batch_x = torch.from_numpy(linspace[:, np.newaxis].astype(np.float32))
        for mc in range(mcsteps):
            with torch.no_grad():
                predictions = model(batch_x)[:, 0]
            all_preds += predictions.numpy() / mcsteps
            new_df = pd.DataFrame(columns=cols, data=list(zip(
                linspace, predictions.numpy(), [name] * len(linspace), [mc] * len(linspace))))

            prediction_df = pd.concat([prediction_df, new_df])


    elif "dropout" in mode:
        model = DropoutNN(hidden=CONFIG_HIDDEN_SIZE)
        optimiser = torch.optim.Adam(model.parameters(), lr=CONFIG_LR, eps=CONFIG_EPS)

        # === TRAIN ===
        with trange(CONFIG_TRAINING_IT) as pbar:
            for _ in pbar:
                optimiser.zero_grad()
                preds = model(batch_x)
                loss = torch.mean(CONFIG_LOSS_CRIT(preds, batch_y))

                loss.backward()
                optimiser.step()
                pbar.set_postfix(loss=loss.detach().numpy())
            pbar.close()

        # === TEST ===
        prediction_df = pd.DataFrame(columns=cols)
        mcsteps = CONFIG_MC_STEPS
        all_preds = np.zeros(len(linspace))
        batch_x = torch.from_numpy(linspace[:, np.newaxis].astype(np.float32))

        for mc in range(mcsteps):
            with torch.no_grad():
                predictions = model(batch_x)[:, 0]  # keep flag: [1, 100] in tensorflow version
            all_preds += predictions.numpy() / mcsteps
            new_df = pd.DataFrame(columns=cols, data=list(zip(
                linspace, predictions.numpy(), [name] * len(linspace), [mc] * len(linspace))))

            prediction_df = pd.concat([prediction_df, new_df])

    elif 'bbb' in mode:

        # Construct model
        n_weight_samples = 5
        bbb_l1 = BBBLayer(1, CONFIG_HIDDEN_SIZE, n_weight_samples)
        bbb_l2 = BBBLayer(CONFIG_HIDDEN_SIZE, 1, n_weight_samples)
        model = nn.Sequential(bbb_l1, nn.ReLU(), bbb_l2)

        # Train model
        optimiser = torch.optim.Adam(model.parameters(), lr=CONFIG_LR, eps=CONFIG_EPS)
        n_epochs = CONFIG_TRAINING_IT

        with trange(n_epochs) as pbar:
            for epoch in pbar:
                optimiser.zero_grad()
                preds = model(batch_x.repeat(n_weight_samples, 1, 1))
                # loss = bbb_criterion(preds, batch_y, [bbb_l1, bbb_l2])
                loss = bbb_criterion(preds, batch_y, [bbb_l1, bbb_l2], crit=CONFIG_LOSS_CRIT)
                loss.backward(retain_graph=(epoch < n_epochs - 1))
                optimiser.step()
            pbar.close()

        # Predict and build dataframe
        prediction_df = pd.DataFrame(columns=cols)
        mcsteps = CONFIG_MC_STEPS
        all_preds = np.zeros(len(linspace))
        batch_x = torch.from_numpy(linspace[:, np.newaxis].astype(np.float32))
        for mc in range(mcsteps // n_weight_samples):
            with torch.no_grad():
                # Note: predictions contains multiple samples because bbb takes multiple weight
                # samples by default.
                predictions = model(batch_x.repeat(n_weight_samples, 1, 1))[:, :, 0]

            for weight_sample_i in range(n_weight_samples):
                sample_predictions = predictions[weight_sample_i]
                all_preds += sample_predictions.numpy() / mcsteps
                new_df = pd.DataFrame(columns=cols, data=list(zip(
                    linspace, sample_predictions.numpy(), [name] * len(linspace), [mc] * len(linspace))))

                prediction_df = pd.concat([prediction_df, new_df])

    elif "ensemble" in mode:
        CONFIG_ADV_ALPHA = config['adv_alpha']
        CONFIG_ADV_EPS = config['adv_eps']
        CONFIG_NUM_NET = config['num_net']

        model = Ensemble(n=CONFIG_NUM_NET, hidden=CONFIG_HIDDEN_SIZE, lr=CONFIG_LR, eps=CONFIG_EPS)
        n_epochs = CONFIG_TRAINING_IT
        batch_x.requires_grad = True

        with trange(n_epochs) as pbar:
            for _ in pbar:
                for model_atom in model.networks:
                    optimiser = model_atom.optimiser
                    optimiser.zero_grad()
                    x_ = model_atom(batch_x)
                    loss = torch.mean(CONFIG_LOSS_CRIT(x_, batch_y))
                    loss.backward(retain_graph=True)

                    # Adversarial training - smooth predictive distribution
                    loss_grads = batch_x.grad.data
                    adv_data = batch_x + CONFIG_ADV_EPS * torch.sign(loss_grads)

                    adv_pred = model_atom(adv_data)
                    adv_loss = torch.mean(CONFIG_LOSS_CRIT(adv_pred, batch_y))
                    optimiser.zero_grad()

                    tot_loss = CONFIG_ADV_ALPHA * loss + (1 - CONFIG_ADV_ALPHA) * adv_loss
                    tot_loss.backward()

                    optimiser.step()
                    pbar.set_postfix(loss=loss.detach().numpy())
            pbar.close()

        # === TEST ===
        prediction_df = pd.DataFrame(columns=cols)
        all_preds = np.zeros(len(linspace))
        batch_x = torch.from_numpy(linspace[:, np.newaxis].astype(np.float32))

        for model_atom in model.networks:
            with torch.no_grad():
                predictions = model_atom(batch_x)[:, 0]
                all_preds += predictions.numpy() / len(model.networks)
                new_df = pd.DataFrame(columns=cols, data=list(zip(
                    linspace, predictions.numpy(), [name] * len(linspace), [len(model.networks)] * len(linspace))))
                prediction_df = pd.concat([prediction_df, new_df])

    else:
        raise NotImplementedError("")

    return prediction_df, weight_dict
