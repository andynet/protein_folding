#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:43:10 2020

@author: andy
"""

# %%
from models import AlphaFold
from datetime import datetime
import numpy as np
import argparse
import random
import torch
import glob


# %%
def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    torch.cuda.manual_seed_all(seed)


# %%
def get_domains(path, n):
    files = glob.glob(path.format('*'))
    domains = [file.split('/')[-1].split('.')[0] for file in files][:n]
    return domains


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
def log(model, train_domains, validation_domains, criterion,
        tensor_path, crop_size, message, device, epoch):

    with torch.no_grad():
        train_loss = evaluate(model, train_domains, criterion, tensor_path, crop_size, device)
        validation_loss = evaluate(model, validation_domains, criterion, tensor_path, crop_size, device)

        print(message.format(
            epoch, model.num_updates, train_loss, validation_loss, datetime.now()),
            flush=True)

    return train_loss, validation_loss


# %%
def evaluate(model, domains, criterion, path, crop_size, device):
    loss = 0
    for domain in domains:
        X, Y = torch.load(path.format(domain))
        i_offset = 0
        j_offset = 0
        X, Y = crop_data(X, Y, i_offset, j_offset, crop_size)
        # print(torch.cuda.memory_stats(device=torch.device('cuda'))['reserved_bytes.all.peak'])
        loss += model.score(X.to(device=device), Y.to(device=device), criterion)
    loss /= len(domains)
    return round(loss.item(), 4)


# %%
def crop_data(X_raw, Y_raw, i_offset, j_offset, crop_size):
    r"""
    Generate inner crops induced by offset.

    Parameters
    ----------
    X_raw : torch.Tensor
        Tensor with shape (C, L, L) and dtype torch.float64
    Y_raw : torch.Tensor
        Tensor with shape (L, L) and dtype torch.int64
    i_offset : int
        upper position of the first crop
    j_offset : int
        left position of the first crop
    crop_size : int
        width and height of the generated crops

    Returns
    -------
    X_cropped : torch.Tensor
        Tensor with shape (N, C, crop_size, crop_size) and dtype torch.float64
        where N is the number of all possible inner crops induced by offset
    Y_cropped : torch.Tensor
        Tensor with shape (N, crop_size, crop_size) and dtype torch.int64
        where N is the number of all possible inner crops induced by offset
    """
    seq_length = X_raw.shape[1]
    i_offsets = list(range(i_offset, seq_length, crop_size))
    j_offsets = list(range(j_offset, seq_length, crop_size))

    if i_offsets[-1] + crop_size > seq_length:
        i_offsets = i_offsets[0:-1]
    if j_offsets[-1] + crop_size > seq_length:
        j_offsets = j_offsets[0:-1]

    N = len(i_offsets) * len(j_offsets)
    X_cropped = torch.zeros((N, X_raw.shape[0], crop_size, crop_size),
                            dtype=torch.float32)
    Y_cropped = torch.zeros((N, crop_size, crop_size),
                            dtype=torch.int64)

    n = 0
    for i in i_offsets:
        for j in j_offsets:
            X_cropped[n, :, :, :] = X_raw[:, i:i + crop_size, j:j + crop_size]
            Y_cropped[n, :, :] = Y_raw[i:i + crop_size, j:j + crop_size]
            n += 1

    return X_cropped, Y_cropped


# %%
def save_model(epoch, train_loss, validation_loss, model, model_path):
    params = f'e_{epoch}_tl_{train_loss}_vl_{validation_loss}'
    torch.save(model.state_dict(), model_path.format(params))


# %%
def train(model, domains, criterion, optimizer,
          path, crop_size, device, batch_size):

    # domains = train_domains; path = args.tensor_path; crop_size = args.crop_size;
    # device = args.device; batch_size = args.batch_size

    # TODO: mode loading of domains out?
    loaded_domains = [torch.load(path.format(x)) for x in domains]
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
    args = argparse.Namespace()
    args.seed = 1
    args.use_cuda = True
    args.tensor_path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors_cs64/{}.pt'
    args.model_path = '/faststorage/project/deeply_thinking_potato/data/prospr/models/{}.pt'

    # memory and time parameters
    args.RESNET_depth = 210              # 220 in prospr, needs to be 210 in our GPU
    args.crop_size = 32                  # 64 in prospr
    args.batch_size = 4                  # 8 in prospr
    args.num_epochs = 640

    # saving parameters
    args.train_limit = 4.1589            # 4.1589 is random model
    args.validation_limit = 4.1589

    # %%
    set_seed(args.seed)

    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    domains = get_domains(args.tensor_path, 40)
    train_domains, validation_domains = data_split(domains, 0.9)

    # %%
    model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
                      RESNET_depth=args.RESNET_depth).to(device=args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    print(f'Total number of parameters: {get_num_params(model)}')

    # %%
    message = '{!s:16}{!s:16}{!s:16}{!s:16}{!s:16}'
    print(message.format(
        'epoch', 'num_updates', 'train_loss', 'validation_loss', 'timestamp'))

    for epoch in range(0, args.num_epochs):
        train_loss, validation_loss = log(model, train_domains, validation_domains,
                                          criterion, args.tensor_path, args.crop_size,
                                          message, args.device, epoch)

        if (train_loss < args.train_limit and validation_loss < args.validation_limit):
            save_model(epoch, train_loss, validation_loss, model, args.model_path)
            args.train_limit = train_loss
            args.validation_limit = validation_loss

        train(model, train_domains, criterion, optimizer,
              args.tensor_path, args.crop_size, args.device, args.batch_size)

    log(model, train_domains, validation_domains,
        criterion, args.tensor_path, args.crop_size, message,
        args.device, args.num_epochs)


# %%
if __name__ == "__main__":
    main()
