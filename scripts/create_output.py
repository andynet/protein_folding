#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:35:51 2020

@author: andyb
"""

# %%
import numpy as np
# import matplotlib.pyplot as plt

# %%
file = '/faststorage/project/deeply_thinking_potato/Data/casp12/training_30'

with open(file, 'r') as f:
    lines = f.readlines()

# %%
for offset in range(0, len(lines), 33):

    identifier = lines[offset + 1].strip()

    sequence = lines[offset + 3].strip()

    PSSM = np.zeros((21, len(sequence)))

    for i in range(0, 21):
        values = lines[offset + 5 + i].strip().split()
        PSSM[i, :] = values

    xs = lines[offset + 27].strip().split()
    ys = lines[offset + 28].strip().split()
    zs = lines[offset + 29].strip().split()

    mask = lines[offset + 31].strip()

    alpha_carbons = []
    for i in range(0, len(sequence)):
        if mask[i] == "+":
            j = i * 3 + 1
            x = float(xs[j])
            y = float(ys[j])
            z = float(zs[j])
            alpha_carbons.append((x, y, z))
        else:
            alpha_carbons.append((np.nan, np.nan, np.nan))

# %%
    distance_matrix = np.zeros((len(sequence), len(sequence)))
    for i in range(len(sequence)):
        for j in range(len(sequence)):
            dist = np.sqrt((alpha_carbons[i][0] - alpha_carbons[j][0])**2
                           + (alpha_carbons[i][1] - alpha_carbons[j][1])**2
                           + (alpha_carbons[i][2] - alpha_carbons[j][2])**2
                           )
            distance_matrix[i, j] = dist

# %%
# plt.imshow(distance_matrix)

    outfile = f'{file}_dir/{identifier}.npy'
    np.save(outfile, distance_matrix)
