#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:44:47 2020

@author: tomasla
"""

# %%
import numpy as np
import torch
import pickle
from torch.nn.functional import softmax
# %% load casp domain ranges and domain lengths
with open('../steps/domain_lengths.pkl', 'rb') as f:
    domain_lengths = pickle.load(f)
    
casp_domain_ranges = np.loadtxt('../steps/casp_domain_ranges.csv', dtype='O', delimiter=',')

# %%
for i in range(len(casp_domain_ranges)):
    t_full = casp_domain_ranges[i][0]
    start, end = casp_domain_ranges[i][1:].astype(np.int)
    target = t_full.split('-')[0]
    L = domain_lengths[target]
    
    if L >= 64:
        P = torch.load(f'../data/our_input/Y_tensors_2020_04_30_model_93/{target}.pred.pt')
        dis, sec, phi, psi = P
    else:
        with open(f'../steps/predicted_outputs/casp13_inception/{target}_out.pkl', 'rb') as f:
            P = pickle.load(f)
        dis, sec, phi, psi = P['distogram'].unsqueeze(0), P['sec'][:, 1:, 0, :], P['phi'][:, :, 0, :], P['psi'][:, :, 0, :]
    
    domain = (
            softmax(dis[:, :, start:end, start:end], dim=1),
            softmax(sec[:, :, start:end], dim=1),
            softmax(phi[:, :, start:end], dim=1),
            softmax(psi[:, :, start:end], dim=1)
        )
    torch.save(domain, f'../steps/casp13_predictions/{t_full}.pred.pt')


# %%  extract sequences
for (d, s, e) in casp_domain_ranges:
    with open(f'../data/our_input/sequences/{d.split("-")[0]}.fasta') as f:
        header = f.readline()
        seq = f.readline()
    
    with open(f'../data/our_input/sequences/{d}.fasta', 'w') as f:
        f.write(header)
        f.write(seq[int(s):int(e)])
        
        
# for target in casp13.loc[:, 'Target']:
    #os.system(f'bash optimize_restarts_casp.sh {target}')