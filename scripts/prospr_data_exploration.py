#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:06:37 2020

@author: tomasla
"""

#%%
# import os 
# os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

import numpy as np

import pickle
import matplotlib.pyplot as plt

def open_pickle(filepath):
    '''Opens ProSPr pickle file and extracts the content'''
    objects = []
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return np.array(objects)

#%% Potts Models
potts = open_pickle('../data/ProSPr/7fd1A00.pkl')

j_matrix = potts[0]['J']
h_matrix = potts[0]['h'] 
frobenius = potts[0]['frobenius_norm']
ijpair = j_matrix[5, 10]
ijpair = ijpair.astype(np.float)
#del potts
#%% PSSMs
pssm = open_pickle('../data/ProSPr/name2pssm.pkl')
pssmxample = pssm[0]['7fd1A00']

del pssm
#%% Distance bins
#bins = open_pickle('../data/ProSPr/name2bins.pkl')
binexample = bins[0]['2qrvD02']

#del bins
#%%
hhblits = open_pickle('../data/ProSPr/name2hh.pkl')
hhexample = hhblits[0]['7fd1A00']

del hhblits
#%%
ss = open_pickle('../data/ProSPr/name2ss.pkl')
ssexample = ss[0]['7fd1A00']

#%%
sequences = open_pickle('../data/ProSPr/name2seq.pkl')
seqexample = sequences[0]['7fd1A00']
#%%
psi = open_pickle('../data/ProSPr/name2psi.pkl')
psiexample = psi[0]['7fd1A00']
#%%
#fig, ax = plt.subplots(1, 4, figsize = (14, 4))
fig = plt.figure(figsize = (14, 4))
ax1 = fig.add_axes([0.03, 0.1, 0.24, 0.85])
ax2 = fig.add_axes([0.3, 0.1, 0.19, 0.85])
ax3 = fig.add_axes([0.52, 0.1, 0.19, 0.85])
ax4 = fig.add_axes([0.74, 0.1, 0.24, 0.85])

j = ax1.imshow(ijpair, cmap = 'viridis', interpolation = 'none')
h = ax2.imshow(h_matrix, cmap = 'viridis')
p = ax3.imshow(pssmxample, cmap = 'viridis')
d = ax4.imshow(binexample, cmap = 'viridis_r')

ax1.set_title('Potts Model: J(i, j)')
ax2.set_title('Potts Model: h-matrix')
ax3.set_title('PSSM')
ax4.set_title('Distance Matrix')

plt.colorbar(j, ax = ax1, fraction = 0.04, orientation = 'horizontal')
plt.colorbar(h, ax = ax2, fraction = 0.04, orientation = 'horizontal')
plt.colorbar(p, ax = ax3, fraction = 0.04, orientation = 'horizontal')
plt.colorbar(d, ax = ax4, fraction = 0.04, orientation = 'horizontal')

plt.savefig('../plots/input_output_colormaps.png', dpi = 200)
#plt.tight_layout()

#%%
# sequence length histogram

seq_length = np.empty(len(sequences[0]))
for i in range(len(sequences[0])):
    seq_length[i] = len(sequences[0][list(sequences[0].keys())[i]])

#%%
plt.hist(seq_length, bins = 100)
plt.title('Distribution of domain lengths')
plt.xlabel('Domain Length')
plt.ylabel('Count')

plt.savefig('../plots/domain_length_distr.png', dpi = 100)

#%% How many distances are missing (proportion)?
bins_missing = {}

for k in bins[0].keys():
    bins_missing[k] = np.mean(bins[0][k] == 0).round(5)    
del k

#%%
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax[0].hist(bins_missing.values(), bins = 50)
ax[1].hist(bins_missing.values(), bins = 150)
ax[1].set_xlim(0, 0.15)

ax[1].set_title('Zoomed in')
fig.suptitle('Proportion of missing distance bins', fontsize = 20)

plt.savefig('../plots/missing_dist_bins.png', dpi = 100)

#%% How many have no missing values?
no_missing = np.sum(np.array(list(bins_missing.values())) == 0)
# returns 15588

#%% which ones are these?
no_missing = []
for k in bins_missing.keys():
    if bins_missing[k] == 0:
        no_missing.append(k)
        
#%%
np.savetxt('../steps/domains_no_missing_dist', np.array(no_missing), fmt = '%s')

