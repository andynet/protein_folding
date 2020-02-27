#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:43:10 2020

@author: andy
"""

# %%
import glob
import torch
import numpy as np
from alphafold_v2 import AlphaFold


# %%
def get_crops(X, Y, indices, crop_size):
    seq_length = X.shape[1]
    crop_middle = crop_size // 2

    batch_in = torch.zeros([len(indices), X.shape[0], crop_size, crop_size],
                           dtype=torch.float64)
    batch_out = torch.zeros([len(indices), crop_size, crop_size],
                            dtype=torch.int64)

    for i, (x, y) in enumerate(indices):
        raw_bounds = [x - crop_middle, x + crop_middle,
                      y - crop_middle, y + crop_middle]

        clipped_bounds = [max(0, raw_bounds[0]), min(seq_length, raw_bounds[1]),
                          max(0, raw_bounds[2]), min(seq_length, raw_bounds[3])]

        image_bounds = [clipped_bounds[0] - x + crop_middle,
                        clipped_bounds[1] - x + crop_middle,
                        clipped_bounds[2] - y + crop_middle,
                        clipped_bounds[3] - y + crop_middle]

        tmp = X[:,
                clipped_bounds[0]:clipped_bounds[1],
                clipped_bounds[2]:clipped_bounds[3]]

        batch_in[i, :,
                 image_bounds[0]:image_bounds[1],
                 image_bounds[2]:image_bounds[3]
                 ] = tmp

        tmp = Y[clipped_bounds[0]:clipped_bounds[1],
                clipped_bounds[2]:clipped_bounds[3]]

        batch_out[i,
                  image_bounds[0]:image_bounds[1],
                  image_bounds[2]:image_bounds[3]
                  ] = tmp

    return batch_in, batch_out


# %%
def fit(model, X, Y, batch_size, crop_size,
        criterion, optimizer):
    """
    Fit the model to the data.

    Parameters
    ----------
    X : torch.Tensor
        Tensor with shape (C, L, L) and dtype torch.float64
    Y : torch.Tensor
        Tensor with shape (L, L) and dtype torch.int64
    batch_size : int
        influences X_batch and Y_batch variable sizes
        X_batch.shape = (batch_size, C, crop_size, crop_size)
        Y_batch.shape = (batch_size, crop_size, crop_size)
    crop_size : int
        influences X_batch and Y_batch variable sizes (see batch_size)
    criterion : torch.nn.modules.loss.CrossEntropyLoss
        a.k.a. loss function
    optimizer : torch.optim.adam.Adam
        SGD method used

    Returns
    -------
    None.

    """

    model.train()
    indices = np.array([
        np.random.choice(np.arange(X.shape[1]), size=24),
        np.random.choice(np.arange(X.shape[1]), size=24)]).T

    for i in range(0, indices.shape[0], batch_size):
        X_batch, Y_batch = get_crops(X, Y,
                                     indices[i:i + batch_size], crop_size)
        optimizer.zero_grad()
        prediction = model.forward(X_batch)
        loss = criterion(prediction, Y_batch)
        loss.backward()
        optimizer.step()
        model.num_updates += 1


# %%
def predict(model, X):
    model.eval()
    X = X.view(-1, X.shape[0], X.shape[1], X.shape[2])
    return model.forward(X)


# %%
def score(model, X, Y, criterion):
    model.eval()
    prediction = predict(model, X)
    Y = Y.view(-1, Y.shape[0], Y.shape[1])
    loss = criterion(prediction, Y)
    return loss


# %%
def evaluate(domains, path, model, criterion):
    loss = 0
    for domain in domains:
        X, Y = torch.load(path.format(domain))
        loss += score(model, X, Y, criterion)
    loss /= len(domains)
    return round(loss.item(), 4)


# %%
def train(model, domains, criterion, path):
    for domain in domains:
        X, Y = torch.load(path.format(domain))
        fit(model, X, Y, batch_size=8, crop_size=64,
            criterion=criterion, optimizer=optimizer)


# %%
model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
                  RESNET_depth=4).to(dtype=torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# %% check number of parameters
# num_params = 0
# for name, param in model.named_parameters():
#     # print(f'{name}\t{param.shape}\t{np.prod(list(param.shape))}')
#     num_params += np.prod(list(param.shape))

# %%
files = glob.glob('/faststorage/project/deeply_thinking_potato/'
                  'data/prospr/tensors2/*.pt')
domains = [file.split('/')[-1].split('.')[0] for file in files]
del files

# %%
train_domains = domains[0:8]        # shuffle this in real run
validation_domains = domains[8:10]  # shuffle this in real run

# %%
message = 'epoch: {}\ttrain_loss: {}\tvalidation_loss: {}'
path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors2/{}.pt'
num_epochs = 1

# %%
for epoch in range(0, num_epochs):
    with torch.no_grad():
        train_loss = evaluate(train_domains, path, model, criterion)
        validation_loss = evaluate(validation_domains, path, model, criterion)
        print(message.format(epoch, train_loss, validation_loss))

    train(model, train_domains, criterion, path)

with torch.no_grad():
    train_loss = evaluate(train_domains, path, model, criterion)
    validation_loss = evaluate(validation_domains, path, model, criterion)
    print(message.format(num_epochs, train_loss, validation_loss))

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
