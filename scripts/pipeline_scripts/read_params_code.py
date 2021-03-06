#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:06:13 2020

@author: tomasla
"""
# %%
import numpy as np
import pickle


def fread(fid, nelements, dtype):
    """Equivalent to Matlab fread function
    
    Made by: JPFrancoia
    https://gist.github.com/JPFrancoia/37246e016e4fba56363423497a8b75b7
    """

    if dtype is np.str:
        dt = np.uint8
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    # data_array.shape = (nelements, 1)

    return data_array


# %%
def read_params(filename, return_neff=False):
    """Reads binary file generated by plmc. Returns J matrix and h_matrix"""

    f_params = open(filename, 'rb')
    params = {}
    # 1: Number of sites
    L = fread(f_params, 1, np.int32)[0]

    # 2: Number of codes in alphabet
    q = fread(f_params, 1, np.int32)[0]

    # 3: Number of valid sequences in alignment
    params['numSeqs'] = fread(f_params, 1, np.int32)[0]

    # 4: Number of invalid sequences in alignment
    params['numInvalidSeqs'] = fread(f_params, 1, np.int32)[0]

    # 5: Number of iterations
    params['numIter'] = fread(f_params, 1, np.int32)[0]

    # 6: Theta
    params['theta'] = fread(f_params, 1, np.float32)[0]

    # 7: lambda for fields (lh)
    params['lambda_h'] = fread(f_params, 1, np.float32)[0]

    # 8: lambda for couplings (le)
    params['lambda_J'] = fread(f_params, 1, np.float32)[0]

    # 9: group lambda for couplings (lg)
    params['lambda_group'] = fread(f_params, 1, np.float32)[0]

    # 10: effective sample size (nEff)
    params['nEff'] = fread(f_params, 1, np.float32)[0]

    # 11: Alphabet
    temp = fread(f_params, q, np.str)
    params['alphabet'] = np.array([chr(i) for i in temp])

    # 12: Sequence number of neighbors (self included)
    params['weights'] = fread(f_params, params['numSeqs'] + params['numInvalidSeqs'], np.float32)

    # 13: Target sequence
    temp = fread(f_params, L, np.str)
    params['target_seq'] = np.array([chr(i) for i in temp])

    # 14: Offset map
    params['offset_map'] = fread(f_params, L, np.int32)

    # 15: Single-site marginals
    params['fi'] = fread(f_params, L * q, np.float32).reshape((L, q))

    # 16: Single-site biases
    params['hi'] = fread(f_params, L * q, np.float32).reshape((L, q))
    params['Jij'] = np.zeros((L, L, q, q), dtype=np.float32)

    # 17: Pairwise marginals
    block = fread(f_params, L * (L - 1) * q * q // 2, np.float32)
    params['fij'] = np.zeros((L, L, q, q), dtype=np.float32)
    offset = 0
    for i in range(L - 1):
        for j in range((i + 1), L):
            params['fij'][j, i, :, :] = block[offset + np.arange(q * q)].reshape((q, q)).T
            params['fij'][i, j, :, :] = block[offset + np.arange(q * q)].reshape((q, q))
            offset = offset + q * q

    # 18: Couplings Jij
    block = fread(f_params, L * (L - 1) * q * q // 2, np.float32)
    params['Jij'] = np.zeros((L, L, q, q))
    offset = 0
    for i in range(L - 1):
        for j in range((i + 1), L):
            params['Jij'][j, i, :, :] = block[offset + np.arange(q * q)].reshape((q, q))
            params['Jij'][i, j, :, :] = block[offset + np.arange(q * q)].reshape((q, q)).T
            offset = offset + q * q

    f_params.close()
    
    if return_neff:
        return params['nEff']
    return params['hi'], params['Jij']
