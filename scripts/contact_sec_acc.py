#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:25:07 2020

@author: tomasla
"""

# %% imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

font = {'size'   : 12}

rc('font', **font)

test_domains = np.loadtxt('../data/our_input/test_domains.csv', dtype='O')
# %%

def contact_map(rrfile):
    with open(rrfile) as rr:
        contacts_rr = rr.readlines()
    
    L = len(contacts_rr[0].strip())
    
    contacts = np.zeros((L, L))
    
    for i in range(1, len(contacts_rr)):
        i, j = contacts_rr[i].strip().split()[:2]
        
        contacts[int(i) - 1, int(j) - 1] = 1
    
    contacts = contacts + contacts.T - np.eye(L)
    
    return contacts
    
# %%
def contacts(predrr, realrr, precision=True):
    
    pred = contact_map(predrr)
    real = contact_map(realrr)
    
    if precision:
        TP = np.sum((pred + real) == 2)
        FP = np.sum((pred + 2 * real) == 1)
        
        return TP / (TP + FP)
    else:  # accuracy
        return np.mean((pred - real) == 0)


# %%
def ss_acc(predss, realss):
    with open(predss) as f:
        f.readline()
        pred = f.readline().strip()
    with open(realss) as f:
        f.readline()
        real = f.readline().strip()
    
    return np.mean([pred[i] == real[i] for i in range(len(pred))])
# %%
contacts_sec = np.empty((500, 3), dtype='O')

for i, domain in enumerate(test_domains):
    cp = contacts(f'../data/our_input/contacts/{domain}.pred.rr', f'../data/our_input/contacts/{domain}.real.rr')
    s = ss_acc(f'../data/our_input/secondary_structures/{domain}.pred.ss', f'../data/our_input/secondary_structures/{domain}.real.ss')
    
    contacts_sec[i] = [domain, cp, s]
    if i % 50 == 0:
        print(f'{i} domains done')
        
# %%
df = pd.DataFrame(contacts_sec[:, 1:], contacts_sec[:, 0], columns=['Contact_Precision', 'Secondary_Accuracy'])
df.index.name = 'Domain'

df.to_csv('../steps/contacts_secondary_accuracy.csv')

# %%
df = pd.read_csv('../steps/contacts_secondary_accuracy.csv', index_col=0)

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.distplot(df.loc[:, 'Contact_Precision'], ax=ax[0], norm_hist=True)
ax[0].axvline(np.mean(df.loc[:, 'Contact_Precision']), ls='--', c='C1', alpha=0.5)
ax[0].text(np.mean(df.loc[:, 'Contact_Precision']) - 0.125 * (1.05 - 0.47), 0.9 * 15.5, f"{np.round(np.mean(df.loc[:, 'Contact_Precision']), 3)}")
ax[0].text(np.mean(df.loc[:, 'Contact_Precision']) - 0.15 * (1.05 - 0.47), 0.85 * 15.5, r"$\pm$" + f"{np.round(np.std(df.loc[:, 'Contact_Precision']), 3)}")
ax[0].set_xlim(0.47, 1.05)
ax[0].set_ylim(0, 15.5)
ax[0].set_xlabel('Contact Precision')
ax[0].set_ylabel('Density')

sns.distplot(df.loc[:, 'Secondary_Accuracy'], ax=ax[1])
ax[1].axvline(np.mean(df.loc[:, 'Secondary_Accuracy']), ls='--', c='C1', alpha=0.5)
ax[1].text(np.mean(df.loc[:, 'Secondary_Accuracy']) - 0.125 * (1.05 - 0.25), 0.9*6.5, f"{np.round(np.mean(df.loc[:, 'Secondary_Accuracy']), 3)}")
ax[1].text(np.mean(df.loc[:, 'Secondary_Accuracy']) - 0.15 * (1.05 - 0.25), 0.85*6.5, r'$\pm$' + f"{np.round(np.std(df.loc[:, 'Secondary_Accuracy']), 3)}")
ax[1].set_xlim(0.25, 1.05)
ax[1].set_ylim(0, 6.5)
ax[1].set_xlabel('Secondary Structure Accuracy (Q3)')
ax[1].set_ylabel('Density')
plt.tight_layout()
plt.savefig('../plots/contacts_secondary_eval.png')
