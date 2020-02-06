#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:06:37 2020

@author: tomasla
"""

#%%
import os 
os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

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

h_matrix = potts[0]['h'] 
frobenius = potts[0]['frobenius_norm']

del potts
#%% PSSMs
pssm = open_pickle('../data/ProSPr/name2pssm.pkl')
pssmxample = pssm[0]['7fd1A00']

del pssm
#%% Distance bins
bins = open_pickle('../data/ProSPr/name2bins.pkl')
binexample = bins[0]['7fd1A00']

del bins
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
plt.imshow(binexample, cmap = 'viridis_r')

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