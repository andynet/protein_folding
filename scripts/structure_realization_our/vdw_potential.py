#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:25:29 2020

@author: tomasla
"""

# %%
from structure import Structure
import torch
from optimize_restarts import optimize_restarts, NLLLoss

# %%
d = optimize_restarts('1z6mA02', '../../steps/predicted_outputs/1z6mA02.out', isdict=True, verbose=10, restarts=1,
                      random_state=0, steric=False, lr=0.05, momentum=0.5)



###################################
# %%
s = Structure('1vmgA00', '../../steps/test_predictions/1vmgA00.pred.pt', normal=True, random_state=0)

# %%
with torch.no_grad():
    dc = s.G(True)
    
distance_map = torch.zeros((len(s.seq), len(s.seq)))

for i in range(len(s.seq) - 1):
    for j in range(i + 1, len(s.seq)):
        distance_map[i, j] = torch.sqrt(torch.sum((dc[i] - dc[j]) ** 2))
# %%
C_vdW = 1.7

def steric_repulsion(dmap):
    sr = 0
    d = ((C_vdW ** 2 - dmap ** 2) ** 2) / C_vdW
    for i in range(len(dmap) - 1):
        for j in range(i + 1, len(dmap)):
            if dmap[i, j] < C_vdW:
                sr += d[i, j]
    return sr

# %%
print(steric_repulsion(distance_map))
    
