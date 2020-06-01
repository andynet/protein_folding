#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:05:10 2020

@author: tomasla
"""

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'size'   : 12}

rc('font', **font)

test_domains = np.loadtxt('../data/our_input/test_domains.csv', dtype='O')


# %%
def calc_tmscore(pred_pdb, real_pdb):
    a = os.popen(f"TMalign {pred_pdb} {real_pdb}")
    for line in a.readlines():
        if line.startswith('TM-score'):
            tmscore = line.split()[1]
            break
    
    return float(tmscore)


def confold_tmscore(domain):
    temp_pred = []
    temp_predreal = []
    
    real = f'../data/pdbfiles/{domain[:4]}.pdb'
    for i in range(1, 6):
        pred = f'../data/our_input/confold_out/{domain}_pred/stage2/{domain}_model{i}.pdb'
        predreal = f'../data/our_input/confold_out/{domain}_real/stage2/{domain}_model{i}.pdb'
        
        temp_pred.append(calc_tmscore(pred, real))
        temp_predreal.append(calc_tmscore(predreal, real))
    
    return domain, max(temp_pred), max(temp_predreal)


# %% 1vq8E01 and 1vq8E02 causess TMalign to crash
tmscores = np.empty((498, 3), dtype='O')

ind = 0
for domain in test_domains:
    
    if domain[:4] != '1vq8':
        tmscores[ind] = confold_tmscore(domain)
        ind += 1
    
    if ind % 50 == 0:
        print(f'TM score of {ind} domains calculated')
        
# %%
# df = pd.DataFrame(tmscores, columns = ['Domain', 'Prediction', 'Ideal_Prediction'])

# df.to_csv('../steps/test_tmscores.csv')

# %%
df = pd.read_csv('../steps/test_tmscores.csv', index_col='Domain').drop('Unnamed: 0', axis=1)

# %%
fig = plt.figure(figsize=(6, 6))

plt.plot(df.iloc[:, 0], df.iloc[:, 1], '.k')
plt.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), '--r')

plt.xlabel('Prediction TM-score')
plt.ylabel('Maximum Theoretical TM-score')

plt.tight_layout()
plt.savefig('../plots/test_tmscore.png')


# %% CASP DOMAINS
