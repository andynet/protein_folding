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
import seaborn as sns
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
casp_domain_ranges = np.loadtxt('../steps/casp_domain_ranges.csv', dtype='O', delimiter=',')

# %%
casp_tmscores = np.empty((18, 2), dtype='O')

ind = 0
for k, (domain, s, e) in enumerate(casp_domain_ranges):
    
    if domain not in ['T1009-D1', 'T0955-D1']:
        print(domain)
        real = f'../data/pdbfiles/{domain}.pdb'
        temp = []
        for i in range(1, 6):
            pred = f'../data/our_input/confold_out/{domain}_pred/stage2/{domain}_model{i}.pdb'
            temp.append(calc_tmscore(pred, real))
            
        casp_tmscores[ind] = [domain, np.max(temp)]
        ind += 1

# %%
caspdf = pd.DataFrame(casp_tmscores[:, 1], casp_tmscores[:, 0], columns = ['TM-Score'])
caspdf.index.name = 'Target'
caspdf.to_csv('../steps/casp13_tmscores.csv')
# %%
casp13_team_tmscores = np.array([
    ['T0954-D1', 0.89, 0.8],
    ['T0958-D1', 0.68, 0.55],
    
    ['T0960-D2', 0.39, 0.41],
    ['T0960-D3', 0.67, 0.66],
    ['T0960-D5', 0.73, 0.69],
    
    ['T0963-D2', 0.39, 0.51],
    ['T0963-D3', 0.71, 0.62],
    ['T0963-D5', 0.77, 0.59],
    ['T0966-D1', 0.39, 0.32],
    
    ['T1003-D1', 0.78, 0.76],
    ['T1005-D1', 0.74, 0.64],
    ['T1008-D1', 0.81, 0.28],
    ['T1016-D1', 0.92, 0.85],
    ], dtype='O')

# %%
a = pd.DataFrame(casp_tmscores[:, 1], casp_tmscores[:, 0])
b = pd.DataFrame(casp13_team_tmscores[:, 1:], casp13_team_tmscores[:, 0])

casp13df = pd.merge(a, b, left_index=True, right_index=True)
casp13df.columns = ['NN+CF2', 'AlphaFold', 'RaptorXContact']
casp13df.index.name = 'Target'

casp13df.to_csv('../steps/casp13_us_plus_AF_RX.csv')

# %%
fig = plt.figure(figsize = (12, 7))

sns.stripplot(casp13df.index, casp13df['NN+CF2'], color = 'C0', size=12, marker='p')
sns.stripplot(casp13df.index, casp13df.AlphaFold, color = 'C1', size=12, marker='s')
sns.stripplot(casp13df.index, casp13df.RaptorXContact, color = 'C2', size=12)

plt.xticks(rotation=45)
# %%
aa = casp13df.reset_index()
df = aa.melt('Target', casp13df.columns, 'Team', 'TM-Score')

# %%
fig = plt.figure(figsize=(12, 7))

sns.barplot('Target', 'TM-Score', hue='Team', data=df)
plt.xticks(rotation=45)

plt.tight_layout()

#plt.savefig('../plots/casp13_comaparison.png')
# %%
#T0963_D1 = []
#s, e = 8, 39
#with open('../data/our_input/confold_out/T0963_pred/stage2/T0963_model1.pdb') as f:
#    for line in f.readlines():
#        if line.startswith('ATOM'):
#            temp = line.strip().split()
#            
#            residue_num = int(temp[4])
#            if residue_num in range(s, e):
#                T0963_D1.append(line)

#with open('../data/our_input/confold_out/T0963_pred/stage2/domains/T0963-D1.pdb', 'w') as f:
#    f.write('HEADER T0963-D1 extracted from T0963 prediction\n')
#    for i in T0963_D1:
#        f.write(i)
#    f.write('TER\nEND\n')
