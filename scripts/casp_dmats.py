#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:15:10 2020

@author: tomasla
"""

# %%
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from confoldpdb_to_distmat import get_dmap
from matplotlib import rc
import sys
sys.path.append('pipeline_scripts')
from read_params_code import read_params
from adjustText import adjust_text
import pickle

font = {'size'   : 15}
rc('font', **font)

test_domains = np.loadtxt('../data/our_input/test_domains.csv', dtype='O')
with open('../steps/domain_lengths.pkl', 'rb') as f:
    domain_lengths = pickle.load(f)

casp13df = pd.read_csv('../steps/casp13_us_plus_AF_RX.csv', index_col=0)
casp_domain_ranges = np.loadtxt('../steps/casp_domain_ranges.csv', dtype='O', delimiter=',')

# %% T1005-D1
def plot_distmaps(target, ax, modelnum=1):

    s, e = casp_domain_ranges[np.where(casp_domain_ranges[:, 0] == target)[0][0], 1:]
    s, e = int(s), int(e)
    #  Real
    y = torch.load(f'../data/casp13_test_data/distance_maps/distance_maps32/{target.split("-")[0]}.pt')
    y = y[s:e, s:e]
    
    # % Pred
    dmap, ss, phi, psi = torch.load(f'../steps/casp13_predictions/{target}.pred.pt')
    
    # 
    yhat = torch.argmax(dmap[0, 1:, : , :], dim=0)
    
    #  Induced from Confold
    yconf = get_dmap(target, modelnum)
    
    #fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    ax[0].imshow(y, cmap='viridis_r')
    ax[1].imshow(yhat, cmap='viridis_r')
    ax[2].imshow(yconf, cmap='viridis_r')
    
# %%
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

plot_distmaps('T0958-D1', ax[0])
plot_distmaps('T1016-D1', ax[1])
for i in range(2):
    for j in range(3):
        plt.setp(ax[i, j].get_xticklabels(), visible=False)
        plt.setp(ax[i, j].get_yticklabels(), visible=False)
        ax[i, j].tick_params(axis='both', which='both', length=0)

ax[0, 0].set_ylabel('T0958-D1')
ax[1, 0].set_ylabel('T1016-D1')

ax[0, 0].set_xlabel('Ground Truth')
ax[0, 1].set_xlabel('Predicted')
ax[0, 2].set_xlabel('Predicted + CONFOLD2')

for j in range(3):
    ax[0, j].xaxis.set_label_position('top')

fig.tight_layout()

#plt.savefig('../plots/casp_distance_maps_test_structures.png')

# %% test msa counts
raw = np.loadtxt('../steps/msa_counts.tsv', dtype='O')

# %%
#msacounts = []
#for domain in test_domains:
#    ind = np.where(raw[:, 0] == f'{domain}.a3m')[0]
#    if len(ind) > 0:
#        msacounts.append([domain, int(raw[ind[0], 1])])

#msacounts = np.array(msacounts, dtype='O')

# %%
#df = pd.DataFrame(msacounts[:, 1], msacounts[:, 0], columns=['MSA_depth'])
#df.index.name= 'Domain'

#df.to_csv('../steps/test_msa_counts.csv')

df = pd.read_csv('../steps/test_msa_counts.csv', index_col=0)
# %%
plt.hist(df.MSA_depth, bins=30)

# %%
tmscores = pd.read_csv('../steps/test_tmscores.csv', index_col='Domain').drop('Unnamed: 0', axis=1)

# %%
dfmerged = pd.merge(df, tmscores, left_index = True, right_index = True)

# %%
plt.plot(dfmerged.MSA_depth, dfmerged.Prediction, '.')
plt.tight_layout()

# %% neff
neff = []

for i, domain in enumerate(test_domains):
    try:
        neff_temp = read_params(f'../data/our_input/temp/{domain}.paramfile', True)
        neff.append([domain, neff_temp])
    except:
        pass
    
    if i % 10 == 0:
        print(f'{i+1} domain done. Found {len(neff)} paramfiles')
neff = np.array(neff, dtype='O')

# %%
neff_df = pd.DataFrame(neff[:, 1], neff[:, 0], columns=['Neff'])
neff_df.index.name = 'Domain'

neff_df.to_csv('../steps/test_neff.csv')

# %%
neff_df = pd.read_csv('../steps/test_neff.csv', index_col=0)
# %%

caspneff = []
for i, domain in enumerate(casp13df.index):
    protein = domain.split("-")[0]

    try:
        neff_temp = read_params(f'../data/our_input/temp/{protein}.paramfile', True)
        caspneff.append([domain, neff_temp])
    except:
        pass
    
    if i % 10 == 0:
        print(f'{i+1} domain done. Found {len(caspneff)} paramfiles')

caspneff = np.array(caspneff, dtype='O')

caspneff_df = pd.DataFrame(caspneff[:, 1], caspneff[:, 0], columns=['Neff'])
caspneff_df.index.name = 'Domain'

caspneff_df.to_csv('../steps/casp_neff.csv')

# %%
caspneff_df = pd.read_csv('../steps/casp_neff.csv', index_col=0)

# %%
casp_merged = pd.merge(casp13df, caspneff_df, left_index=True, right_index=True)
tm_neff = pd.merge(neff_df, tmscores, left_index = True, right_index = True)

# %%
casp_dr = pd.DataFrame(casp_domain_ranges[:, 1:], casp_domain_ranges[:, 0])
casp_length = []
for d in casp_merged.index:
    casp_length.append(int(casp_dr.loc[d][1]) - int(casp_dr.loc[d][0]))

test_lengths = []
for i in tm_neff.index:
    test_lengths.append(domain_lengths[i])
# %%
casp_merged['Domain_Length'] = casp_length
tm_neff['Domain_Length'] = test_lengths
# %%
casp_merged.to_csv('../steps/casp_full_results.csv')
tm_neff.to_csv('../steps/test_full_results.csv')

# %%
fig = plt.figure(figsize=(12, 7))

# Test 
plt.plot(tm_neff.Neff, tm_neff.Prediction, '.', alpha=0.6)

# CASP
plt.plot(casp_merged.Neff, casp_merged.loc[:, 'NN+CF2'], '.', markersize=15, alpha=0.8)
#plt.annotate(xy=(casp_merged.Neff, casp_merged.loc[:, 'NN+CF2']), s=casp_merged.index)
#for i in range(len(casp_merged)):
#    plt.annotate(casp_merged.index[i], (casp_merged.iloc[i, 3], casp_merged.iloc[i, 0]))

#texts = []
#for (s, x, y) in np.array(casp_merged.loc[:, ['Neff', 'NN+CF2']].reset_index()):
#    texts.append(plt.text(x, y, s))

#adjust_text(texts, x=x, y=y, autoalign='y',
#            only_move={'points':'y', 'text':'y'}, force_points=0.15,
#            arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.xlabel('Neff')
plt.ylabel('TM-Score')
