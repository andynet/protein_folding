#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:13:11 2020

@author: tomasla

File should contain a pipeline that takes a name of domain, indices (i, j) and
returns concatenated features: J(i, j), (h(i), h(j)), FN(i, j), (P(i), P(j)), (S(i), S(j))  
"""
#%%
# import os 
# os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

import numpy as np
import pandas as pd
import torch
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

# %%
def seq_to_dummy_tensor(seq):
    order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
             'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']

    a = np.array([aa for aa in seq])
    b = pd.get_dummies(a)
    b = b.loc[:, order]

    return b.values


# %%
def output_to_dummy_tensor(output):
    L = output.shape[0]
    result = np.zeros((L, L, 64))
    for i in range(L):
        for j in range(L):
            k = output[i, j]
            result[i, j, k] = 1

    return result


# %%
def open_data(domains):
    tensors = {}
    for domain in domains:
        filepath = f'../data/potts/{domain}.pkl'
        tensors[domain] = open_pickle(filepath)[0]

    sequences = open_pickle('../data/ProSPr/name2seq.pkl')[0]
    seqs = {key: value for key, value in sequences.items() if key in domains}

    for domain in domains:
        tensors[domain]['seq'] = seq_to_dummy_tensor(seqs[domain])

    output = open_pickle('../data/ProSPr/name2bins.pkl')[0]
    for domain in domains:
        tensors[domain]['output'] = output[domain]#output_to_dummy_tensor(output[domain])

    return tensors

#%% Sequence encoding

# Function should take a sequence as input and return it one-hot encoded columns
# i and j

#def ohe_seq_ij(domain, i, j):
#    '''Returns tuple of one hot encoded aminoacids at position i and j in the sequence'''
#    
#    keys = np.array(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'])
          # These keys were also used by ProSPr in the exact order          

#    sequences = open_pickle('../data/ProSPr/name2seq.pkl')
#    seq = sequences[0][domain]
#    
#    aa_i, aa_j = np.zeros(21), np.zeros(21) 
#    aa_i[np.where(keys == seq[i])] = 1
#    aa_j[np.where(keys == seq[j])] = 1
    
#    return torch.from_numpy(aa_i).float().view(1, 21), torch.from_numpy(aa_j).float().view(1, 21)

#%% Extract table J(i, j), columns h(i), h(j) and FN(i, j)

def potts_input(domain, i, j):
    '''Returns tuple of matrix J(i, j), h(i), h(j) and FN(i, j)'''

    potts = open_pickle('../data/potts/' + domain + '.pkl')

    J = potts[0]['J']
    #h = potts[0]['h'] 
    #FN = potts[0]['frobenius_norm']
    
    J_ij = J[i, j]#.astype(np.float)
    
    return torch.from_numpy(J_ij).float()#, h[i], h[j], FN[i, j]

#%% Extract columns i, j from Potts Model

# PSSM file is huge and it does not make sense to have it open everytime 
    # the function is called

#pssms = open_pickle('../data/ProSPr/name2pssm.pkl')

#def pssm_input(domain, i, j):
#    '''Returns tuple of PSSM(i) and PSSM(j)'''
#    
    #pssms = open_pickle('../data/ProSPr/name2pssm.pkl')
#    pssm = pssms[0][domain]
    
#    return pssm[i], pssm[j]

#%% Create 64*64 batches from the J matrix for a given protein

    