#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:36:58 2020

@author: andyb
"""

# %%
from model_aux import AlphaFold, Domains
import numpy as np
import pandas as pd
import random
import torch
import yaml
import glob
from datetime import datetime


# %%
def get_domains(path, n='all'):
    files = glob.glob(path.format('*'))
    domains = [file.split('/')[-1].split('_')[0] for file in files]
    if n == 'all':
        return domains
    else:
        return domains[:n]


# %%
def construct_crop(domains, i, j, crop_size, is_inner, N):
    # print(f'Constructing crop {i} {j} with {N} proteins.')

    n_channels = domains[0][0].shape[1]
    X = torch.zeros((N, n_channels, crop_size, crop_size), dtype=torch.float32)

    dist = torch.zeros((N, crop_size, crop_size), dtype=torch.int64)

    ss_i = torch.zeros((N, crop_size), dtype=torch.int64)
    phi_i = torch.zeros((N, crop_size), dtype=torch.int64)
    psi_i = torch.zeros((N, crop_size), dtype=torch.int64)

    ss_j = torch.zeros((N, crop_size), dtype=torch.int64)
    phi_j = torch.zeros((N, crop_size), dtype=torch.int64)
    psi_j = torch.zeros((N, crop_size), dtype=torch.int64)

    n = 0
    for k, (X_raw, Y_raw) in enumerate(domains):
        if is_inner[k]:

            X[n, :, :, :] = X_raw[0, :, i:i + crop_size, j:j + crop_size]
            dist_raw, ss_raw, phi_raw, psi_raw = Y_raw

            dist[n, :, :] = dist_raw[0, i:i + crop_size, j:j + crop_size]

            ss_i[n, :] = ss_raw[0, i:i + crop_size]
            phi_i[n, :] = phi_raw[0, i:i + crop_size]
            psi_i[n, :] = psi_raw[0, i:i + crop_size]

            ss_j[n, :] = ss_raw[0, j:j + crop_size]
            phi_j[n, :] = phi_raw[0, j:j + crop_size]
            psi_j[n, :] = psi_raw[0, j:j + crop_size]

            n += 1

    Y = (dist, ss_i, phi_i, psi_i, ss_j, phi_j, psi_j)
    return X, Y


# %%
def train(model, loaded_domains, criterion, optimizer,
          crop_size, device, batch_size):

    lengths = [X.shape[3] for (X, _) in loaded_domains]
    max_length = max(lengths)

    i_offset = random.randint(0, crop_size)
    j_offset = random.randint(0, crop_size)

    for i in range(i_offset, max_length, crop_size):
        for j in range(j_offset, max_length, crop_size):

            is_inner = [(max(i, j) + crop_size <= x) for x in lengths]
            N = sum(is_inner)

            if N < batch_size:
                continue

            X, Y = construct_crop(loaded_domains, i, j, crop_size, is_inner, N)

            indices = np.random.permutation(range(N))
            batch_starts = [x * batch_size for x in range(0, N // batch_size)]

            for k in batch_starts:
                X_batch = X[indices[k:k + batch_size], :, :, :].to(device)

                Y_batch = tuple([
                    Y[l][indices[k:k + batch_size]].to(device) for l in range(7)
                ])

                model.fit(X_batch,
                          Y_batch,
                          criterion=criterion, optimizer=optimizer)


# %%
def evaluate(model, loaded_domains, criterion, crop_size, device):
    with torch.no_grad():
        losses = []
        lengths = [X.shape[3] for (X, _) in loaded_domains]
        max_length = max(lengths)

        i_offset = 0
        j_offset = 0

        for i in range(i_offset, max_length, crop_size):
            for j in range(j_offset, max_length, crop_size):

                is_inner = [(max(i, j) + crop_size <= x) for x in lengths]
                N = sum(is_inner)

                if N == 0:
                    continue

                X, Y = construct_crop(loaded_domains, i, j, crop_size, is_inner, N)

                X = X.to(device)
                Y = tuple([Y[l].to(device) for l in range(7)])

                loss = 0.0
                loss = model.score(X,
                                   Y,
                                   criterion=criterion).item()

                losses.append(loss)

    return round(sum(losses) / len(losses), 4)


# %%
config_file = "alphafold_aux.yml"

# %%
cfg = yaml.load(open(config_file), Loader=yaml.FullLoader)

# %%
random.seed(cfg['seed'])
torch.manual_seed(cfg['seed'])
np.random.seed(seed=cfg['seed'])
torch.cuda.manual_seed_all(cfg['seed'])

# %%
train_domains = pd.read_csv(cfg['available_domains'], header=None).loc[:, 0]
X_domains = get_domains(cfg['X_files'].format(domain='*'))
Y_domains = get_domains(cfg['Y_files'].format(domain='*'))
domains = sorted(list(set(X_domains) & set(Y_domains) & set(train_domains)))
random.shuffle(domains)
del train_domains, X_domains, Y_domains

# %%
# cfg['chunk_size'] = 256
# domains = domains[0:cfg['max_domains']]
# chunks = [domains[x:x + cfg['chunk_size']] for x in range(0, len(domains), cfg['chunk_size'])]

# %%
domains = domains[0:cfg['max_domains']]
dataset = Domains(domains, cfg['X_files'], cfg['Y_files'], verbosity=0)
loader = torch.utils.data.DataLoader(dataset, num_workers=cfg['num_workers'])

loaded_data = []
loaded_domains = []
for i, (X, Y, domain) in enumerate(loader):
    length = X.shape[2]
    if length >= cfg['crop_size']:
        loaded_data.append((X, Y))
        loaded_domains.append((domain, length))

# %%
train_loaded = loaded_data[0:int(len(loaded_data) * cfg['train_proportion'])]
validation_loaded = loaded_data[int(len(loaded_data) * cfg['train_proportion']):]

# %%
if cfg['use_cuda'] and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %%
model = AlphaFold(
    in_channels=cfg['in_channels'],
    dist_channels=cfg['dist_channels'],
    ss_channels=cfg['ss_channels'],
    phi_channels=cfg['phi_channels'],
    psi_channels=cfg['psi_channels'],
    up_channels=cfg['up_channels'],
    down_channels=cfg['down_channels'],
    RESNET_depth=cfg['RESNET_depth'],
    crop_size=cfg['crop_size'],
    weights=cfg['weights']
).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']),
                             weight_decay=float(cfg['l2']))

# average mistake should be 53.2606

# %%
message = '{!s:16}{!s:16}{!s:16}{!s:16}{!s:16}'
print(message.format(
    'epoch', 'num_updates', 'train_loss', 'validation_loss', 'timestamp'))

# %%
for epoch in range(0, cfg['max_epochs']):

    train_loss = evaluate(model, train_loaded[0:50], criterion,
                          cfg['crop_size'], device)

    validation_loss = evaluate(model, validation_loaded[0:50], criterion,
                               cfg['crop_size'], device)

    # TODO: if the validation error was worse than last 3 validation errors
    # reload data

    print(message.format(epoch, model.num_updates, train_loss,
                         validation_loss, datetime.now()), flush=True)

    torch.save(model.state_dict(), cfg['model_path'].format(epoch=epoch))

    train(model, train_loaded, criterion, optimizer,
          cfg['crop_size'], device, cfg['batch_size'])

train_loss = evaluate(model, train_loaded[0:50], criterion,
                      cfg['crop_size'], device)

validation_loss = evaluate(model, validation_loaded[0:50], criterion,
                           cfg['crop_size'], device)

print(message.format(epoch, model.num_updates, train_loss,
                     validation_loss, datetime.now()), flush=True)

torch.save(model.state_dict(), cfg['model_path'].format(epoch=epoch))
