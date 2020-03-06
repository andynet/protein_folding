#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:44:32 2020

Perceptron - Most basic model

@author: tomasla
"""

# %% Imports
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# %% Constants
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.1
EPOCHS = 50
DOMAINS_EVAL = 10

# %% Load domain names
domains = []
for filename in os.listdir('/faststorage/project/deeply_thinking_potato/data/prospr/tensors3/'):
    domains.append(filename.split('.')[0])
del filename
domains = np.array(domains)

# %% Pick Train and Validation Domains
np.random.seed(25)

TRAIN_SIZE = int((1 - VALIDATION_RATIO - TEST_RATIO) * len(domains))
TEST_SIZE = int(TEST_RATIO * len(domains))

test_ind = np.random.choice(np.arange(len(domains)), TEST_SIZE, replace=False)

train_val_ind = np.setdiff1d(np.arange(len(domains)), test_ind)
train_ind = np.random.choice(train_val_ind, TRAIN_SIZE, replace=False)
val_ind = np.setdiff1d(train_val_ind, train_ind)

assert len(train_ind) + len(val_ind) + len(test_ind) == len(domains)

train_domains = domains[train_ind]
validation_domains = domains[val_ind]
test_domains = domains[test_ind]


# %% Perceptron Class
class Perceptron(torch.nn.Module):
    def __init__(self, input_size=675):
        super(Perceptron, self).__init__()
        
        self.bn0 = torch.nn.BatchNorm1d(input_size)
        self.hidden1 = torch.nn.Linear(input_size, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.hidden2 = torch.nn.Linear(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.linear = torch.nn.Linear(64, 64)

    def IO_from_indices(self, X, Y, indices):
        """Create Input and Output correspondent to the ij pairs in indices"""

        Inp = torch.empty(len(indices), X.shape[0])
        Out = torch.empty(len(indices))

        for counter, (i, j) in enumerate(indices):
            Inp[counter] = X[:, i, j]
            Out[counter] = Y[i, j]
        return Inp.to(**kwargs), Out.to(dtype=torch.long, device=device)

    def forward(self, X):
        X = self.bn0(X)
        #X = F.relu(self.hidden1(X))
        #X = self.bn1(X)
        #X = F.relu(self.hidden2(X))
        #X = self.bn2(X)
        return F.log_softmax(self.hidden1(X), dim=1)

    def fit(self, X, Y, batch_size, optimizer):

        self.train()
        batch_loops = X.shape[1]**2 // batch_size

        for b in range(batch_loops):
            indices = np.array([
                np.random.choice(np.arange(X.shape[1]), size=batch_size),
                np.random.choice(np.arange(X.shape[1]), size=batch_size)]).T
            X_batch, Y_batch = self.IO_from_indices(X, Y, indices)

            optimizer.zero_grad()
            prediction = self.forward(X_batch)
            # loss = criterion(prediction, Y_batch)
            loss = F.nll_loss(prediction, Y_batch)
            loss.backward()
            optimizer.step()

    def score(self, X, Y):
        self.eval()
        X, Y = X.to(**kwargs), Y.to(dtype=torch.long, device=device)
        prediction = self.forward(X)
        # loss = criterion(prediction, Y)
        loss = F.nll_loss(prediction, Y)
        return loss

    def evaluate(self, domains, path):
        loss = 0
        for domain in domains:
            X, Y = torch.load(path.format(domain))
            X, Y = X.view(675, X.shape[1]**2).to(**kwargs).t(), Y.view(X.shape[1]**2).to(dtype=torch.long, device=device)
            loss += self.score(X, Y)
        loss /= len(domains)
        return round(loss.item(), 4)


def train(epochs, model, train_domains, validation_domains, optimizer):

    message = 'epoch: {}\ttrain_loss: {}\tvalidation_loss: {}'
    path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors3/{}.pt'

    print_size = min(50, len(train_domains))
    print_frequency = np.ceil(len(train_domains) / print_size)

    model.history = np.empty((epochs, 3))
    
    DOMAINS_EVAL0 = min(len(train_domains), DOMAINS_EVAL)
    train_eval_ind = np.random.choice(np.arange(len(train_domains)), DOMAINS_EVAL0, replace=False)
    val_eval_ind = np.random.choice(np.arange(len(validation_domains)), DOMAINS_EVAL0, replace=False)
    train_eval, val_eval = train_domains[train_eval_ind], validation_domains[val_eval_ind]

    for epoch in range(epochs):
        with torch.no_grad():
            train_loss = model.evaluate(train_eval, path)
            validation_loss = model.evaluate(val_eval, path)
            model.history[epoch] = [epoch, train_loss, validation_loss]
            print('\n', message.format(epoch, train_loss, validation_loss))

        if epoch % 5 == 0:
            np.savetxt('/faststorage/project/deeply_thinking_potato/steps/perceptron_history.csv', model.history, delimiter=',')
        for i, domain in enumerate(train_domains):
            X, Y = torch.load(path.format(domain))
            model.fit(X, Y, batch_size=100, optimizer=optimizer)

            if i % print_frequency == 0:
                m = f'Trained: {round(100 * i / len(train_domains), 2)}%' + ' ' + '#' * int(i / print_frequency + 1)
                sys.stdout.write('\r' + m)
        m = f'Trained: 100%' + ' ' + '#' * int(i / print_frequency + 2)
        sys.stdout.write('\r' + m)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'device': device, 'dtype': torch.float64}
print(device)

model = Perceptron(input_size=675).to(**kwargs)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
train(500, model, train_domains[0:1], validation_domains, optimizer)

# %%
X_, Y_ = torch.load('/faststorage/project/deeply_thinking_potato/data/prospr/tensors3/{}.pt'.format(train_domains[0]))
X_ = X_.view(675, X_.shape[1]**2).reshape(-1, 675)

out0 = model(X_.to(**kwargs)).cpu().detach().numpy()

# %%
dist = np.empty(out0.shape[0])

for i in range(out0.shape[0]):
    dist[i] = np.argmax(out0[i])
    
# %%
dist = dist.reshape((64, 64))