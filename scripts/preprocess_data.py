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


# %%
def onehot_encode(s):
    '''Convert a sequence into 21xN matrix. For dataloader'''
    seq_profile = np.zeros((len(s), 21), dtype=np.short)
    aa_order = 'ARNDCQEGHILKMFPSTWYVX'
    seq_order = [aa_order.find(letter) for letter in s]

    for i, letter in enumerate(seq_order):
        seq_profile[i, letter] = 1
    return seq_profile


# %% filepaths
# inputs
# https://files.physics.byu.edu/data/prospr/dicts/name2pssm.pkl
name2pssm_pkl = '/faststorage/project/deeply_thinking_potato/data/ProSPr/name2pssm.pkl'
# https://files.physics.byu.edu/data/prospr/dicts/name2hh.pkl
name2hh_pkl = '/faststorage/project/deeply_thinking_potato/data/ProSPr/name2hh.pkl'
# https://files.physics.byu.edu/data/prospr/dicts/name2seq.pkl
name2seq_pkl = '/faststorage/project/deeply_thinking_potato/data/ProSPr/name2seq.pkl'
# https://files.physics.byu.edu/data/prospr/potts/
potts_template_pkl = '/faststorage/project/deeply_thinking_potato/data/potts/{domain}.pkl'

# output
# https://files.physics.byu.edu/data/prospr/dicts/name2bins.pkl
name2bins_pkl = '/faststorage/project/deeply_thinking_potato/data/ProSPr/name2bins.pkl'

# %%
files = glob.glob('/faststorage/project/deeply_thinking_potato/data/potts/*.pkl')
domains = [file.split('/')[-1].split('.')[0] for file in files]
del files

# %%
# 15400ms
name2pssm = pickle.load(open(name2pssm_pkl, "rb"))
name2hh = pickle.load(open(name2hh_pkl, "rb"))
name2seq = pickle.load(open(name2seq_pkl, "rb"))
name2bins = pickle.load(open(name2bins_pkl, "rb"))
del name2pssm_pkl, name2hh_pkl, name2seq_pkl, name2bins_pkl

# assert len(name2pssm) == len(name2hh) == len(name2seq) == len(name2bins) == len(domains)

# %%
# 321ms
domain = domains[0]

potts = pickle.load(open(potts_template_pkl.format(domain=domain), "rb"))
potts_J, potts_h, potts_frobenius_norm, potts_score = potts.values()
del potts

seq_1hot = onehot_encode(name2seq[domain])
del name2seq

pssm = name2pssm[domain]
del name2pssm

hh = name2hh[domain]
del name2hh

bins = name2bins[domain]
del name2bins

# %%
L = seq_1hot.shape[0]

# %%
# 117ms
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

del potts_J, potts_score, potts_frobenius_norm, tile_i, tile_j, \
    potts_h_i, potts_h_j, seq_1hot_i, seq_1hot_j, pssm_i, pssm_j, \
    hh_i, hh_j, seq_len

# %%
in_tensor = torch.tensor(in_tensor)
out_tensor = torch.tensor(bins)

# %%
torch.save([in_tensor, out_tensor], f'{domain}.pt')

# %%
in_tensor, out_tensor = torch.load('4n7rC02.pt')

# %%
INPUT_LAYERS = 613
INPUT_DIMENSION = 32

BINS = 64
CROP_SIZE = 32

EPOCHS = 1
EVAL_DOMAINS = 1

i, j = 50, 50
len_seq = 111


lower_i = max(0, i - 32)
upper_i = min(len_seq, i + 32)
irange = upper_i - lower_i

lower_j = max(0, j - 32)
upper_j = min(len_seq, j + 32)
jrange = upper_j - lower_j

xi = min(32, i)
yi = min(32, len_seq - i)

xj = min(32, j)
yj = min(32, len_seq - j)


# translate pixels correctly! this is necessary because we map the frame to the zeroed np array!
vlower_i = lower_i - i + 32
vupper_i = upper_i - i + 32

vlower_j = lower_j - j + 32
vupper_j = upper_j - j + 32

# %%
# np.unravel_index(np.argmax(potts_J, axis=None), potts_J.shape)
# tmp = potts_J.reshape((111, 111, 484))
# in_tensor = np.zeros((INPUT_LAYERS, INPUT_DIMENSION, INPUT_DIMENSION))

# %%
# in_tensor = np.zeros((700, 1000, 1000))

# %%

# %%
# potts_J 484 = 22x22
# potts_score 1
# potts_fn 1
# tile i 1
# tile j 1
# potts_h i 22
# potts_h j 22
# seq_profile_crop i 21
# seq_profile_crop j 21
# pssm i 20
# pssm j 20
# hh i 30
# hh j 30
# seq_len 1
# ------------
# together 679


# %%


# %%
# def andyho_magia(domain, batch_indices):
#     pass


# %%
# batch_indices = np.array([(1, 1), (50, 50)])

# def get_entry()
# result = np.zeros()

# result[0, :, :] = 5

# %%
