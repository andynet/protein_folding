#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:42:11 2020

@author: tomasla
"""
# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

font = {'size'   : 12}

rc('font', **font)

# %%
x = torch.linspace(-3, 3, 300)

def hardtanh(x):
    res = []
    for i in x:
        if i < -1:
            res.append(-1)
        elif i > 1:
            res.append(1)
        else:
            res.append(i)
    return res

def dhardtanx(x):
    res = []
    for i in x:
        if torch.abs(i) > 1:
            res.append(0)
        else:
            res.append(1)
    return res

def delu(x):
    res = []
    for i in x:
        if i < 0:
            res.append(np.exp(i))
        else:
            res.append(1)
    return res
# %%
fig, ax = plt.subplots(3, 4, figsize=(15, 9))

# xaxis
for i in range(3):
    for j in range(4):
        ax[i, j].plot(x, [0 for i in range(300)], '--k', alpha=0.2)
        ax[i, j].plot([0 for i in range(300)], x,'--k', alpha=0.2)
        
        ax[i, j].set_xlim(-3, 3)
        ax[i, j].set_ylim(-3, 3)
        
# Identity
ax[0, 0].plot(x, x, 'C0')
ax[0, 1].plot(x, [1 for i in range(300)], 'C0')
ax[0, 0].set_title(r"$f(x) = x$")
ax[0, 1].set_title(r"$f'(x) = 1$")

# tanh
ax[0, 2].plot(x, torch.tanh(x), 'C1')
ax[0, 3].plot(x, 1 - torch.tanh(x) ** 2, 'C1')
ax[0, 2].set_title(r"$f(x) = tanh(x)$")
ax[0, 3].set_title(r"$f'(x) = 1 - tanh(x)^2$")

# sigmoid
ax[1, 0].plot(x, torch.sigmoid(x), 'C1')
ax[1, 1].plot(x, torch.sigmoid(x)*(1 - torch.sigmoid(x)), 'C1')
ax[1, 0].set_title(r"$f(x) = sigmoid(x)$")
ax[1, 1].set_title(r"$f'(x) = sigmoid(x)*(1 - sigmoid(x))$")

# hardtanh
ax[1, 2].plot(x, hardtanh(x), 'C0')
ax[1, 3].plot(x, dhardtanx(x), 'C0')
ax[1, 2].set_title(r"$f(x) = hardtanh(x)$")
ax[1, 3].set_title(r"$f'(x) = \{1~if~|x| \leq 1, 0~otherwise\}$")

# relu
ax[2, 0].plot(x, torch.relu(x), 'C0')
ax[2, 1].plot(x, torch.relu(x) / x, 'C0')
ax[2, 0].set_title(r"$f(x) = relu(x)$")
ax[2, 1].set_title(r"$f'(x) = \{0~if~x < 0, 1~otherwise\}$")

# elu
ax[2, 2].plot(x, torch.nn.functional.elu(x), 'C1')
ax[2, 3].plot(x, delu(x), 'C1')
ax[2, 2].set_title(r"$f(x) = elu(x)$")
ax[2, 3].set_title(r"$f'(x) = \{e^x~if~x < 0, 1~otherwise\}$")

fig.tight_layout()

plt.savefig('../plots/activation_functions_plus_derivatives.png')

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 9))

# xaxis
for i in range(2):
    for j in range(3):
        ax[i, j].plot(x, [0 for i in range(300)], '--k', alpha=0.2)
        ax[i, j].plot([0 for i in range(300)], x,'--k', alpha=0.2)
        
        ax[i, j].set_xlim(-3, 3)
        ax[i, j].set_ylim(-3, 3)
        
# Identity
ax[0, 0].plot(x, x, 'k')
ax[0, 0].set_title(r"$f(x) = x$")

# tanh
ax[0, 1].plot(x, torch.tanh(x), 'k')
ax[0, 1].set_title(r"$f(x) = tanh(x)$")

# sigmoid
ax[0, 2].plot(x, torch.sigmoid(x), 'k')
ax[0, 2].set_title(r"$f(x) = sigmoid(x)$")

# hardtanh
ax[1, 0].plot(x, hardtanh(x), 'k')
ax[1, 0].set_title(r"$f(x) = hardtanh(x)$")

# relu
ax[1, 1].plot(x, torch.relu(x), 'k')
ax[1, 1].set_title(r"$f(x) = relu(x)$")

# elu
ax[1, 2].plot(x, torch.nn.functional.elu(x), 'k')
ax[1, 2].set_title(r"$f(x) = elu(x)$")

fig.tight_layout()

plt.savefig('../plots/activation_functions.png')