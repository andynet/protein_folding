#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:43:10 2020

@author: andy
"""

# %%
from models import AlphaFold
from datetime import datetime
import pickle
import numpy as np
import argparse
import random
import psutil
import torch
import glob


# %%
def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    torch.cuda.manual_seed_all(seed)


# %%
def get_domains(path, n='all'):
    files = glob.glob(path.format('*'))
    domains = [file.split('/')[-1].split('.')[0] for file in files]
    if n == 'all':
        return domains
    else:
        return domains[:n]


# %%
def data_split(domains, proportion):
    train_size = int(len(domains) * proportion)
    random.shuffle(domains)
    return domains[:train_size], domains[train_size:]


# %%
def get_num_params(model):
    num_params = 0
    for name, param in model.named_parameters():
        num_params += np.prod(list(param.shape))
        # print(name, param.shape, np.prod(list(param.shape)))
    return num_params


# %%
def evaluate(model, loaded_domains, criterion, crop_size, device):
    with torch.no_grad():
        losses = []
        lengths = [Y.shape[0] for (_, Y) in loaded_domains]
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

                loss = 0.0
                loss = model.score(X.to(device=device),
                                   Y.to(device=device),
                                   criterion=criterion).item()

                losses.append(loss)

    return round(sum(losses) / len(losses), 4)


# %%
def save_model(epoch, train_loss, validation_loss, model, model_path):
    params = f'e_{epoch}_tl_{train_loss}_vl_{validation_loss}'
    torch.save(model.state_dict(), model_path.format(params))


# %%
def train(model, loaded_domains, criterion, optimizer,
          crop_size, device, batch_size):

    # domains = train_domains; path = args.tensor_path; crop_size = args.crop_size;
    # device = args.device; batch_size = args.batch_size

    lengths = [Y.shape[0] for (_, Y) in loaded_domains]
    max_length = max(lengths)

    i_offset = random.randint(0, crop_size)
    j_offset = random.randint(0, crop_size)

    for i in range(i_offset, max_length, crop_size):
        for j in range(j_offset, max_length, crop_size):

            is_inner = [(max(i, j) + crop_size <= x) for x in lengths]
            N = sum(is_inner)

            X, Y = construct_crop(loaded_domains, i, j, crop_size, is_inner, N)

            indices = np.random.permutation(range(N))
            batch_starts = [x * batch_size for x in range(0, N // batch_size)]

            for k in batch_starts:
                X_batch = X[indices[k:k + batch_size], :, :, :]
                Y_batch = Y[indices[k:k + batch_size], :, :]

                model.fit(X_batch.to(device=device),
                          Y_batch.to(device=device),
                          criterion=criterion, optimizer=optimizer)


# %%
def construct_crop(domains, i, j, crop_size, is_inner, N):
    X = torch.zeros((N, 675, crop_size, crop_size),
                    dtype=torch.float32)
    Y = torch.zeros((N, crop_size, crop_size),
                    dtype=torch.int64)

    n = 0
    for k, (X_raw, Y_raw) in enumerate(domains):
        if is_inner[k]:
            X[n, :, :, :] = X_raw[:, i:i + crop_size, j:j + crop_size]
            Y[n, :, :] = Y_raw[i:i + crop_size, j:j + crop_size]
            n += 1

    return X, Y


# %%
def main():
    # %%
    args = argparse.Namespace()
    args.seed = 1
    args.use_cuda = True
    args.tensor_path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors_cs64/{}.pt'
    args.model_path = '/faststorage/project/deeply_thinking_potato/data/prospr/models/{}.pt'

    args.train_path = '/faststorage/project/deeply_thinking_potato/data/prospr/dicts/TRAIN-p_names.pkl'
    args.validation_path = '/faststorage/project/deeply_thinking_potato/data/prospr/dicts/VS-p_names.pkl'
    # args.test_path = '/faststorage/project/deeply_thinking_potato/data/prospr/dicts/TEST-p_names.pkl'

    # memory and time parameters
    args.RESNET_depth = 200              # 220 in prospr, needs to be 210 in our GPU
    args.crop_size = 64                  # 64 in prospr
    args.batch_size = 8                  # 8 in prospr
    args.num_epochs = 1000

    # saving parameters
    args.train_limit = 4.1589            # 4.1589 is random model
    args.validation_limit = 4.1589

    # %%
    set_seed(args.seed)

    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # %%
    train_domains = pickle.load(open(args.train_path, "rb"))
    validation_domains = pickle.load(open(args.validation_path, "rb"))

    available_domains = get_domains(args.tensor_path)

    # %%
    train_domains = list(set(train_domains) & set(available_domains))
    validation_domains = list(set(validation_domains) & set(available_domains))

    # %%
    model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
                      RESNET_depth=args.RESNET_depth).to(device=args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    print(f'Total number of parameters: {get_num_params(model)}')

    message = '{!s:16}{!s:16}{!s:16}{!s:16}{!s:16}'
    print(message.format(
        'epoch', 'num_updates', 'train_loss', 'validation_loss', 'timestamp'))

    # %%
    train_loaded = []
    i = 0
    while psutil.virtual_memory().percent < 60:
        x = train_domains[i]
        train_loaded.append(torch.load(args.tensor_path.format(x)))
        i += 1

    # %%
    validation_loaded = []
    j = 0
    while psutil.virtual_memory().percent < 70:
        x = validation_domains[j]
        validation_loaded.append(torch.load(args.tensor_path.format(x)))
        j += 1

    # %%
    for epoch in range(0, args.num_epochs):

        train_loss = evaluate(model, train_loaded[0:50], criterion,
                              args.crop_size, args.device)

        validation_loss = evaluate(model, validation_loaded[0:50], criterion,
                                   args.crop_size, args.device)

        print(message.format(epoch, model.num_updates, train_loss,
                             validation_loss, datetime.now()), flush=True)

        if (train_loss < args.train_limit and validation_loss < args.validation_limit):
            save_model(epoch, train_loss, validation_loss, model, args.model_path)
            args.train_limit = train_loss
            args.validation_limit = validation_loss

        train(model, train_loaded, criterion, optimizer,
              args.crop_size, args.device, args.batch_size)

    train_loss = evaluate(model, train_loaded[0:50], criterion,
                          args.crop_size, args.device)

    validation_loss = evaluate(model, validation_loaded[0:50], criterion,
                               args.crop_size, args.device)

    print(message.format(epoch, model.num_updates, train_loss,
                         validation_loss, datetime.now()), flush=True)

    if (train_loss < args.train_limit and validation_loss < args.validation_limit):
        save_model(epoch, train_loss, validation_loss, model, args.model_path)
        args.train_limit = train_loss
        args.validation_limit = validation_loss


# %%
if __name__ == "__main__":
    main()
