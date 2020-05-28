#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:33:39 2020

@author: tomasla
"""

# %%
from Bio.PDB import PDBParser, protein_letters_3to1
import numpy as np
import matplotlib.pyplot as plt
import torch

# %%
def get_dmap(target):
    path = f'../data/our_input/confold_out/{target}_pred/stage2/{target}_model1.pdb'
    with open(path) as f:
        pdb = f.readlines()
    
    coords = []
    for i in pdb:
        if i.startswith('ATOM'):
            line = i.split()
            atom, aa = line[2], line[3]
            x, y, z = float(line[5]), float(line[6]), float(line[7])
            if aa == 'GLY':
                if atom == 'CA':
                    coords.append([x, y, z])
            else:
                if atom == 'CB':
                    coords.append([x, y, z])
    coords = np.array(coords)
    
    
    L = len(coords)
    dmap = np.zeros((L, L))
    
    for i in range(L-1):
        for j in range(i + 1, L):
            dmap[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
    dmap = dmap + dmap.T
    
    bins = np.concatenate(([0], np.linspace(2, 22, 31)))
    dmap = np.digitize(dmap, bins)
    return torch.from_numpy(dmap)

# %%
#fig, ax = plt.subplots(5, 1)
#for i, t in enumerate(['4i4tE01', '1vmgA00', '4mveA00', '2l1iA00', '3m4yA01']):
#for i, t in enumerate(['1vmgA00', '4mveA00', '2l1iA00']):
#    dmap = get_dmap(t)
#    torch.save(dmap, f'../data/our_input/confold_out/distance_maps/{t}_dmap.pt')
#    ax[i].imshow(dmap, cmap='viridis_r')