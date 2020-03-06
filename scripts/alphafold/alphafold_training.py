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
import gc


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
def train(model, domains, criterion, path, crop_size, device, batch_size):
    for domain in domains:
        X, Y = torch.load(path.format(domain))
        seq_length = X.shape[2]

        i_offset = random.randint(0, min(crop_size, seq_length - crop_size))
        j_offset = random.randint(0, min(crop_size, seq_length - crop_size))
        X_input, Y_input = crop_data(X, Y, i_offset, j_offset, crop_size)

        while X_input.shape[0] < batch_size:
            i_offset = random.randint(0, min(crop_size, seq_length - crop_size))
            j_offset = random.randint(0, min(crop_size, seq_length - crop_size))
            X_crops, Y_crops = crop_data(X, Y, i_offset, j_offset, crop_size)
            X_input = torch.cat([X_input, X_crops], 0)
            Y_input = torch.cat([Y_input, Y_crops], 0)

        model.fit(X_input.to(device=device),
                  Y_input.to(device=device),
                  batch_size=batch_size,
                  criterion=criterion, optimizer=optimizer)


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
def get_num_params(model):
    num_params = 0
    for name, param in model.named_parameters():
        num_params += np.prod(list(param.shape))
        # print(name, param.shape, np.prod(list(param.shape)))
    return num_params


# %%
def data_split(domains, proportion):
    train_size = int(len(domains) * proportion)
    random.shuffle(domains)
    return domains[:train_size], domains[train_size:]


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
def save_model(epoch, train_loss, validation_loss, model, model_path):
    params = f'e_{epoch}_tl_{train_loss}_vl_{validation_loss}'
    torch.save(model.state_dict(), model_path.format(params))


# %%
args = argparse.Namespace()
tensor_path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors_cs64/{}.pt'
model_path = '/faststorage/project/deeply_thinking_potato/data/prospr/models/{}.pt'
args.use_cuda = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.memory_summary(device=device, abbreviated=True))

if args.use_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


# %%
set_seed(0)
domains = get_domains(tensor_path, 2)
train_domains, validation_domains = data_split(domains, 0.9)

# crop_size = 64
# num_epochs = 210
# batch_size = 8

crop_size = 32
num_epochs = 64
batch_size = 4

train_limit = 4.1589        # because this should be the random loss
validation_limit = 4.1589

# %%
model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
                  RESNET_depth=210)
model = model.to(device=args.device, dtype=torch.float32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
# print(f'Total number of parameters: {get_num_params(model)}')

# %%
message = '{!s:16}{!s:16}{!s:16}{!s:16}{!s:16}'
print(message.format(
    'epoch', 'num_updates', 'train_loss', 'validation_loss', 'timestamp'))

for epoch in range(0, num_epochs):
    train_loss, validation_loss = log(model, train_domains, validation_domains,
                                      criterion, tensor_path, crop_size, message,
                                      args.device, epoch)

    if train_loss < train_limit and validation_loss < validation_limit:
        save_model(epoch, train_loss, validation_loss, model, model_path)
        best_train_loss = train_loss
        best_validation_loss = validation_loss

    train(model, train_domains, criterion, tensor_path, crop_size, args.device, batch_size)

# print(torch.cuda.memory_summary(device=args.device, abbreviated=True))
log(model, train_domains, validation_domains,
    criterion, tensor_path, crop_size, message,
    args.device, num_epochs)

# %%
# del model

# # %%
# gc.collect()
# torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=device, abbreviated=True))
# print(torch.cuda.memory_stats(device=device)['allocated_bytes.all.current'])
# print(torch.cuda.memory_stats(device=device)['active_bytes.all.current'])
# print(torch.cuda.memory_stats(device=device)['reserved_bytes.all.current'])

# %%
# .to(device=args.device)

# this should be used to create tensors?
# torch.Tensor.new_empty(self, size)