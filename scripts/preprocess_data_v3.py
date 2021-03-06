#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:42:26 2020

@author: andyb

inspired by https://github.com/dellacortelab/prospr/blob/\
    0726dac9fe0316d217c674b2e0e4f09059fb2405/prospr/dataloader.py
"""

# input.shape = (CROP_SIZE**2, 1, INPUT_LAYERS, INPUT_DIMENSION, INPUT_DIMENSION)
# output.shape = (CROP_SIZE**2, 1)

# %% imports
import numpy as np
import pickle
import glob
import torch
import os


# %%
def onehot_encode(s):
    '''Convert a sequence into 21xN matrix. For dataloader'''
    seq_profile = np.zeros((len(s), 21), dtype=np.short)
    aa_order = 'ARNDCQEGHILKMFPSTWYVX'
    seq_order = [aa_order.find(letter) for letter in s]

    for i, letter in enumerate(seq_order):
        seq_profile[i, letter] = 1
    return seq_profile


# %%
def get_tensors(domain, potts_template_pkl, name2seq, name2pssm, name2hh, name2bins):

    potts = pickle.load(open(potts_template_pkl.format(domain=domain), "rb"))
    potts_J, potts_h, potts_frobenius_norm, potts_score = potts.values()

    seq_1hot = onehot_encode(name2seq[domain])

    pssm = name2pssm[domain]

    hh = name2hh[domain]

    bins = name2bins[domain]

    L = seq_1hot.shape[0]

    potts_J = potts_J.reshape((L, L, 484))
    potts_score = potts_score.reshape((L, L, 1))
    potts_frobenius_norm = potts_frobenius_norm.reshape((L, L, 1))
    tile_i = np.repeat(np.arange(0, L)[:, np.newaxis], L, axis=1).reshape((L, L, 1))
    tile_j = np.repeat(np.arange(0, L)[np.newaxis, :], L, axis=0).reshape((L, L, 1))
    potts_h_i = np.repeat(potts_h[0:L, np.newaxis, :], L, axis=1)
    potts_h_j = np.repeat(potts_h[np.newaxis, 0:L, :], L, axis=0)
    seq_1hot_i = np.repeat(seq_1hot[0:L, np.newaxis, :], L, axis=1)
    seq_1hot_j = np.repeat(seq_1hot[np.newaxis, 0:L, :], L, axis=0)
    pssm_i = np.repeat(pssm[0:L, np.newaxis, :], L, axis=1)
    pssm_j = np.repeat(pssm[np.newaxis, 0:L, :], L, axis=0)
    hh_i = np.repeat(hh[0:L, np.newaxis, :], L, axis=1)
    hh_j = np.repeat(hh[np.newaxis, 0:L, :], L, axis=0)
    seq_len = np.full((L, L, 1), L)

    arrays = [potts_J, potts_score, potts_frobenius_norm, tile_i, tile_j,
              potts_h_i, potts_h_j, seq_1hot_i, seq_1hot_j, pssm_i, pssm_j,
              hh_i, hh_j, seq_len]

    in_tensor = np.concatenate(arrays, axis=2)

    in_tensor = torch.tensor(in_tensor).permute(2, 0, 1).to(dtype=torch.float64)
    out_tensor = torch.tensor(bins).to(dtype=torch.int64)

    return in_tensor, out_tensor


# %% filepaths
# inputs
# https://files.physics.byu.edu/data/prospr/dicts/name2pssm.pkl
name2pssm_pkl = \
    '/faststorage/project/deeply_thinking_potato/data/prospr/dicts/name2pssm.pkl'
# https://files.physics.byu.edu/data/prospr/dicts/name2hh.pkl
name2hh_pkl = \
    '/faststorage/project/deeply_thinking_potato/data/prospr/dicts/name2hh.pkl'
# https://files.physics.byu.edu/data/prospr/dicts/name2seq.pkl
name2seq_pkl = \
    '/faststorage/project/deeply_thinking_potato/data/prospr/dicts/name2seq.pkl'
# https://files.physics.byu.edu/data/prospr/potts/
potts_template_pkl = \
    '/faststorage/project/deeply_thinking_potato/data/prospr/potts/{domain}.pkl'

# output
# https://files.physics.byu.edu/data/prospr/dicts/name2bins.pkl
name2bins_pkl = \
    '/faststorage/project/deeply_thinking_potato/data/prospr/dicts/name2bins.pkl'

# %%
#files = glob.glob('/faststorage/project/deeply_thinking_potato/data/prospr/potts/*.pkl')
#domains = [file.split('/')[-1].split('.')[0] for file in files]
#del files

# This is much faster
domains = []
for filename in os.listdir('/faststorage/project/deeply_thinking_potato/data/prospr/potts/'):
    domains.append(filename.split('.')[0])
 
domains_noNA = np.loadtxt('/faststorage/project/deeply_thinking_potato/steps/domains_no_missing_dist', dtype = 'O')

# %%
# 15400ms
name2pssm = pickle.load(open(name2pssm_pkl, "rb"))
name2hh = pickle.load(open(name2hh_pkl, "rb"))
name2seq = pickle.load(open(name2seq_pkl, "rb"))
name2bins = pickle.load(open(name2bins_pkl, "rb"))
del name2pssm_pkl, name2hh_pkl, name2seq_pkl, name2bins_pkl

# assert len(name2pssm) == len(name2hh) \
#     == len(name2seq) == len(name2bins) == len(domains)

# %%
# this is needed because some dictionaries do not contain all domains
domains = set(domains)\
    .intersection(domains_noNA)\
    .intersection(name2pssm.keys())\
    .intersection(name2hh.keys())\
    .intersection(name2seq.keys())\
    .intersection(name2bins.keys())

# 15318 domains
#domains = sorted(list(domains))[0:10]
# %%
def make_pt(domain):
    data = get_tensors(domain, potts_template_pkl,
                       name2seq, name2pssm, name2hh, name2bins)

    if data[0].shape[1] == data[0].shape[2] == data[1].shape[0] == data[1].shape[1]:
        output_file = \
            f'/faststorage/project/deeply_thinking_potato/data/prospr/tensors3/{domain}.pt'
        torch.save(data, output_file)
        print(output_file)
    else:
        print(f'Domain {domain} is damaged.')

# %%
for domain in domains:
    data = get_tensors(domain, potts_template_pkl,
                       name2seq, name2pssm, name2hh, name2bins)

    if data[0].shape[1] == data[0].shape[2] == data[1].shape[0] == data[1].shape[1]:
        output_file = \
            f'/faststorage/project/deeply_thinking_potato/data/prospr/tensors3/{domain}.pt'
        torch.save(data, output_file)
        print(output_file)
    else:
        print(f'Domain {domain} is damaged.')
        
# %% There was something wrong with domain 2179 - 1qqfA00
# what was wrong? do we really need these few domains for the price 
# of decreasing the consistency of the data?
for i in range(2180, len(domains)):
    make_pt(list(domains)[i])

# %% And also with domain 3472 - 1vavA00
for i in range(3473, len(domains)):
    make_pt(list(domains)[i])

# %% and domain 3968 - 1uwkA02
for i in range(3969, len(domains)):
    make_pt(list(domains)[i])

# %% and domain 8499 - 5d6eA02
for i in range(8500, len(domains)):
    make_pt(list(domains)[i])
    
# %% and domain 10225 - 1kwgA01
for i in range(10226, len(domains)):
    make_pt(list(domains)[i])
    
# %% and domain 11101 - 3besL00
for i in range(11102, len(domains)):
    make_pt(list(domains)[i])
    
# %% and domain 11414 - 1i1rA01
for i in range(11415, len(domains)):
    make_pt(list(domains)[i])

# %% and domain 13680 - 1wyhA00
for i in range(13681, len(domains)):
    make_pt(list(domains)[i])
        
# DONE
