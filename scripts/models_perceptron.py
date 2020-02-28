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
import glob
# %%


class Perceptron(torch.nn.Module):
    def __init__(self, input_size=675):
        super(Perceptron, self).__init__()
        self.linear = torch.nn.Linear(input_size, 64)

    def IO_from_indices(self, X, Y, indices):
        """Create Input and Output correspondent to the ij pairs in indices"""

        Inp = torch.empty(len(indices), X.shape[0])
        Out = torch.empty(len(indices))

        for counter, (i, j) in enumerate(indices):
            Inp[counter] = X[:, i, j]
            Out[counter] = Y[i, j]
        return Inp.to(dtype=torch.float64), Out.to(dtype=torch.long)

    def forward(self, X):
        return F.log_softmax(self.linear(X), dim=1)

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
        prediction = self.forward(X)
        # loss = criterion(prediction, Y)
        loss = F.nll_loss(prediction, Y)
        return loss

    def evaluate(self, domains, path):
        loss = 0
        for domain in domains:
            X, Y = torch.load(path.format(domain))
            X, Y = X.view(675, X.shape[1]**2).to(dtype=torch.float64).t(), Y.view(X.shape[1]**2).to(dtype=torch.long)
            loss += self.score(X, Y)
        loss /= len(domains)
        return round(loss.item(), 4)


def train(epochs, model, train_domains, validation_domains, optimizer):
    
    message = 'epoch: {}\ttrain_loss: {}\tvalidation_loss: {}'
    path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors2/{}.pt'
    
    for epoch in range(5):
        with torch.no_grad():
            train_loss = model.evaluate(train_domains, path)
            validation_loss = model.evaluate(validation_domains, path)
            print(message.format(epoch, train_loss, validation_loss))
            
        for domain in domains:
            X, Y = torch.load(path.format(domain))
            model.fit(X, Y, batch_size=100, optimizer=optimizer)
# %%
#domains = np.loadtxt('../steps/domains_no_missing_dist', dtype = 'O')
files = glob.glob('/faststorage/project/deeply_thinking_potato/'
                  'data/prospr/tensors2/*.pt')
domains = [file.split('/')[-1].split('.')[0] for file in files]
del files
train_domains = domains[0:8]        # shuffle this in real run
validation_domains = domains[8:10]

model = Perceptron(input_size=675).to(dtype=torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()
# %%
train(5, model, train_domains, validation_domains, optimizer)
# %%
#for epoch in range(5):
#    with torch.no_grad():
#        train_loss = model.evaluate(train_domains, path)
#        validation_loss = model.evaluate(validation_domains, path)
#        print(message.format(epoch, train_loss, validation_loss))
#    train(model, train_domains, path, optimizer)
