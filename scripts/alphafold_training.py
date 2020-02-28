#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:43:10 2020

@author: andy
"""

# %%
from alphafold_v2 import AlphaFold
from datetime import datetime
import numpy as np
import random
import torch
import glob


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
    i_offsets = list(range(i_offset, seq_length, crop_size))[0:-1]
    j_offsets = list(range(j_offset, seq_length, crop_size))[0:-1]

    N = len(i_offsets) * len(j_offsets)
    X_cropped = torch.zeros((N, X_raw.shape[0], crop_size, crop_size),
                            dtype=torch.float64)
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
def evaluate(model, domains, criterion, path, crop_size):
    loss = 0
    for domain in domains:
        X, Y = torch.load(path.format(domain))
        i_offset = 0
        j_offset = 0
        X, Y = crop_data(X, Y, i_offset, j_offset, crop_size)
        loss += model.score(X, Y, criterion)
    loss /= len(domains)
    return round(loss.item(), 4)


# %%
def train(model, domains, criterion, path, crop_size):
    for domain in domains:
        X, Y = torch.load(path.format(domain))
        print(domain, evaluate(model, [domain], criterion, path, crop_size), X.shape)
        i_offset = random.randint(0, crop_size - 1)
        j_offset = random.randint(0, crop_size - 1)
        X, Y = crop_data(X, Y, i_offset, j_offset, crop_size)
        model.fit(X, Y, batch_size=8, criterion=criterion, optimizer=optimizer)
        print(domain, evaluate(model, [domain], criterion, path, crop_size))


# %%
model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
                  RESNET_depth=4).to(dtype=torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# %% check number of parameters
# num_params = 0
# for name, param in model.named_parameters():
#     print(f'{np.prod(list(param.shape))}\t{param.shape}\t{name}')
#     num_params += np.prod(list(param.shape))
# print(f'{num_params}\ttotal')

# %%
files = glob.glob('/faststorage/project/deeply_thinking_potato/'
                  'data/prospr/tensors2/*.pt')
domains = [file.split('/')[-1].split('.')[0] for file in files]
del files

# %%
train_domains = domains[0:8]        # shuffle this in real run
validation_domains = domains[8:]    # shuffle this in real run

# %%
message = 'epoch: {}\ttrain_loss: {}\tvalidation_loss: {}\ttime: {}'
path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors2/{}.pt'
num_epochs = 5
crop_size = 32

# %%
for epoch in range(0, num_epochs):
    with torch.no_grad():
        train_loss = evaluate(model, train_domains, criterion, path, crop_size)
        validation_loss = evaluate(model, validation_domains, criterion, path, crop_size)
        print(message.format(epoch, train_loss, validation_loss, datetime.now()))

    train(model, train_domains, criterion, path, crop_size)

with torch.no_grad():
    train_loss = evaluate(model, train_domains, criterion, path, crop_size)
    validation_loss = evaluate(model, validation_domains, criterion, path, crop_size)
    print(message.format(num_epochs, train_loss, validation_loss, datetime.now()))

# %%
path = '/faststorage/project/deeply_thinking_potato/data/prospr/model.pt'
# torch.save(model.state_dict(), path)

# %%
# model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
#                   RESNET_depth=4)
# model.load_state_dict(torch.load(path))

# %%
# total loss = 10*distance_loss + sec_struct_loss1 + sec_struct_loss2
# + torsion_angle_loss1..4

# 500 000 iterations
# batch size = 8
