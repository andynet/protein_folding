#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:05:21 2020

@author: tomasla
"""

# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
from matplotlib import rc
import seaborn as sns
font = {'size'   : 12}

rc('font', **font)

test_domains = np.loadtxt('../data/our_input/test_domains.csv', dtype='O')
with open('../steps/domain_lengths.pkl', 'rb') as f:
    domain_lengths = pickle.load(f)
# %%
def nll_d(d, Y):
    """NLL loss for the distograms"""
    l = torch.empty((len(Y) ** 2))
    for i in range(len(Y)):
        for j in range(len(Y)):
            l[i * len(Y) + j] = - torch.log(d[0, Y[i, j], i, j])
    return torch.mean(l).item()

def nll_a(a, Y):
    """NLL loss for auxiliary targets"""
    l = torch.empty((len(Y)))
    for i in range(len(Y)):
        l[i] = - torch.log(a[0, Y[i], i])
    return torch.mean(l).item()

# %%
test_losses = np.empty((500, 5), dtype='O')

for i, d in enumerate(test_domains):
    p = torch.load(f'../steps/test_predictions/{d}.pred.pt')
    y = torch.load(f'../data/our_input/Y_tensors/{d}_Y.pt')
    
    dist = nll_d(p[0], y[0])
    sec = nll_a(p[1], y[1].to(torch.long) -  1)
    phi = nll_a(p[2], y[2])
    psi = nll_a(p[3], y[3])
    
    test_losses[i] = (d, dist, sec, phi, psi)
    
    if i % 10 == 0:
        print(f'{i} domains done')

# %%
tl_df = pd.DataFrame(test_losses[:, 1:], test_losses[:, 0], columns=['NLL_DG', 'NLL_SEC', 'NLL_PHI', 'NLL_PSI'])
tl_df.index.name = 'Domain'

tl_df.to_csv('../steps/test_losses.csv')

# %%
tl_df = pd.read_csv('../steps/test_losses.csv', index_col=0)

# %%
test_lengths = []
for i in range(500):
    test_lengths.append(domain_lengths[tl_df.index[i]])

# %%
tl_df['Domain_length'] = test_lengths

tl_df['NLL'] = 10 * tl_df.NLL_DG + tl_df.NLL_SEC + tl_df.NLL_PHI + tl_df.NLL_PSI

# %%
fig = plt.figure(figsize=(12, 5))
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[3, 1])

ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])

ax0.plot(tl_df.Domain_length, tl_df.NLL, '.', alpha=0.7, c='navy')
ax1.hist(tl_df.NLL, orientation='horizontal', bins=50, density=True, edgecolor='white', color='navy')

ax0.set_xlabel('Domain Length')
ax0.set_ylabel('NLL loss')
ax1.set_xlabel('Density')
ax1.set_ylabel('NLL loss')

ax0.axhline(-(10 * np.log(1/32) + 2 * np.log(1/9) + 4 * np.log(1/37)), ls='--', c='k', alpha=0.3)
ax1.axhline(-(10 * np.log(1/32) + 2 * np.log(1/9) + 4 * np.log(1/37)), ls='--', c='k', alpha=0.3)

fig.tight_layout()

plt.savefig('../plots/test_losses_distributions.png')