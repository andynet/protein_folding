#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:35:51 2020

@author: andyb
"""

# %%
import numpy as np
import pandas as pd
import os


# %%
def get_PSSM(lines, seq_length, offset):
    result = np.zeros((21, seq_length))
    for i in range(0, 21):
        values = lines[offset + 5 + i].strip().split()
        result[i, :] = values

    return result


# %%
def get_alpha_carbons(lines, offset, seq_length, mask):
    result = []

    xs = lines[offset + 27].strip().split()
    ys = lines[offset + 28].strip().split()
    zs = lines[offset + 29].strip().split()

    for i in range(0, len(sequence)):
        if mask[i] == "+":
            j = i * 3 + 1
            x = float(xs[j])
            y = float(ys[j])
            z = float(zs[j])
            result.append((x, y, z))
        else:
            result.append((np.nan, np.nan, np.nan))

    return result


# %%
def get_distance_matrix(alpha_carbons, seq_length):
    distance_matrix = np.zeros((len(sequence), len(sequence)))
    for i in range(len(sequence)):
        for j in range(len(sequence)):
            dist = np.sqrt((alpha_carbons[i][0] - alpha_carbons[j][0])**2
                           + (alpha_carbons[i][1] - alpha_carbons[j][1])**2
                           + (alpha_carbons[i][2] - alpha_carbons[j][2])**2
                           )
            distance_matrix[i, j] = dist

    return distance_matrix


# %%
def get_PSSM_wider(PSSM):
    L = PSSM.shape[1]
    W = PSSM.shape[0]
    result = np.zeros((L, L, W))
    for i in range(0, L):
        for j in range(0, L):
            for k in range(0, W):
                result[i, j, k] = (PSSM[k, i] + PSSM[k, j]) / 2

    return result


# %%
shared_domains = '/faststorage/project/deeply_thinking_potato/steps/prospr_vs_proteinnet12_shared_domains.csv'
df = pd.read_csv(shared_domains)

names = df['DF_name'].unique()
for name in names:

    file = f'/faststorage/project/deeply_thinking_potato/data/casp12/{name}'
    with open(file, 'r') as f:
        lines = f.readlines()

    domains = df.query(f'DF_name == "{name}"')['Domain']
    for domain in domains:

        for offset in range(0, len(lines), 33):
            identifier = lines[offset + 1].strip()
            if identifier == domain:
                break

        sequence = lines[offset + 3].strip()
        PSSM = get_PSSM(lines, len(sequence), offset)
        mask = lines[offset + 31].strip()
        alpha_carbons = get_alpha_carbons(lines, offset, len(sequence), mask)

        os.makedirs(f'{file}_dir', mode=0o777, exist_ok=True)

        distance_matrix = get_distance_matrix(alpha_carbons, len(sequence))
        outfile = f'{file}_dir/{identifier}_dist.npy'
        print(outfile)
        np.save(outfile, distance_matrix)

        PSSM_wider = get_PSSM_wider(PSSM)
        outfile = f'{file}_dir/{identifier}_PSSM_wider.npy'
        print(outfile)
        np.save(outfile, PSSM_wider)

# %%
# import matplotlib.pyplot as plt
# plt.imshow(distance_matrix)
