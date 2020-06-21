#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:25:07 2020

@author: tomasla
"""

# %% imports
import pandas as pd
import numpy as np
import torch
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
def contacts(predrr, realrr=None, real=None, precision=True):
    
    pred = contact_map(predrr)
    
    if real is None:
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
def casp_eval(predrr, realcm, predss, realsec):
    
    # contact map precision
    predcm = contact_map(predrr)
    TP, FP = 0, 0
    for i in range(len(predcm)):
        for j in range(len(predcm)):
            if realcm[i, j] > -1:
                if predcm[i, j] == 1:
                    if realcm[i, j] == 1:
                        TP += 1
                    else:
                        FP += 1
    precision = TP / (TP + FP)
    
    # secondary structure accuracy
    with open(predss) as f:
        f.readline()
        predsec = f.readline().strip()
    
    correct = 0
    for i in range(len(predsec)):
        if realsec[i] != 'X':
            if predsec[i] == realsec[i]:
                correct += 1
    accuracy = correct / len(predsec)
    
    return precision, accuracy

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

HISTCOL = 'C0'
EDGE = 'black'
MEANCOL = 'black'
RANDCOL = 'C1'
BINS = 32
ALPHA = 0.8

ax[0].hist(df.loc[:, 'Contact_Precision'], density=True, bins=BINS, color=HISTCOL, edgecolor=EDGE, alpha=ALPHA)
ax[0].axvline(np.mean(df.loc[:, 'Contact_Precision']), ls='--', c=MEANCOL, alpha=0.5)
ax[0].text(np.mean(df.loc[:, 'Contact_Precision']) - 0.122 * (1.05 - 0.47), 0.9 * 15.5, f"{np.round(np.mean(df.loc[:, 'Contact_Precision']), 3)}")
ax[0].text(np.mean(df.loc[:, 'Contact_Precision']) - 0.15 * (1.05 - 0.47), 0.85 * 15.5, r"$\pm$" + f"{np.round(np.std(df.loc[:, 'Contact_Precision']), 3)}")
ax[0].set_xlim(0.47, 1.05)
ax[0].set_ylim(0, 15.5)
ax[0].set_xlabel('Contact Precision')
ax[0].set_ylabel('Density')
ax[0].axvline(0.5, ls='--', c=RANDCOL)

ax[1].hist(df.loc[:, 'Secondary_Accuracy'], density=True, bins=BINS, color=HISTCOL, edgecolor=EDGE, alpha=ALPHA)
ax[1].axvline(np.mean(df.loc[:, 'Secondary_Accuracy']), ls='--', c=MEANCOL, alpha=0.5)
ax[1].text(np.mean(df.loc[:, 'Secondary_Accuracy']) - 0.122 * (1.05 - 0.25), 0.9*6.5, f"{np.round(np.mean(df.loc[:, 'Secondary_Accuracy']), 3)}")
ax[1].text(np.mean(df.loc[:, 'Secondary_Accuracy']) - 0.15 * (1.05 - 0.25), 0.85*6.5, r'$\pm$' + f"{np.round(np.std(df.loc[:, 'Secondary_Accuracy']), 3)}")
ax[1].set_xlim(0.25, 1.05)
ax[1].set_ylim(0, 6.5)
ax[1].set_xlabel('Secondary Structure Accuracy (Q3)')
ax[1].set_ylabel('Density')
ax[1].axvline(0.33, ls='--', c=RANDCOL)
plt.tight_layout()

plt.savefig('../plots/contacts_secondary_eval.png')

# %% CASP
casp_domain_ranges = np.loadtxt('../steps/casp_domain_ranges.csv', dtype='O', delimiter=',')
casp13df = pd.read_csv('../steps/casp13_us_plus_AF_RX.csv', index_col=0)

# %% create real contact maps
HIGH_BIN = 11
def get_cmap_secondary(target, return_coords=False):
    
    
    with open(f'../data/our_input/secondary_structures/{target}.real.ss') as f:
        sec = f.readline().strip()

    secondary = []
    
    path = f'../data/pdbfiles/{target}.pdb'
    with open(path) as f:
        pdb = f.readlines()
    
    coords = []
    ind = 0  # residue ind
    for i in pdb:
        if i.startswith('ATOM'):
            line = i.split()
            atom, aa = line[2], line[3]
            
            rind = int(line[4])
            
            
            x, y, z = float(line[5]), float(line[6]), float(line[7])
            
            if aa == 'GLY':
                if atom == 'CA':
                    if ind > 0:
                        rind_previous = coords[-1][0]
                        missing = rind - rind_previous - 1
                        for k in range(missing):
                            coords.append([rind_previous + k + 1, 0, 0, 0])
                            secondary.append('X')
                            
                    coords.append([rind, x, y, z])
                    secondary.append(sec[ind])
                    ind += 1
            else:
                if atom == 'CB':
                    if ind > 0:
                        rind_previous = coords[-1][0]
                        missing = rind - rind_previous - 1
                        for k in range(missing):
                            coords.append([rind_previous + k + 1, 0, 0, 0])
                            secondary.append('X')
                    coords.append([rind, x, y, z])
                    secondary.append(sec[ind])
                    ind += 1
    coords = np.array(coords)[:, 1:]
    
    if return_coords:
        return coords
    
    L = len(coords)
    dmap = np.zeros((L, L))
    
    for i in range(L-1):
        for j in range(i + 1, L):
            if np.sum(coords[i] == 0) == 3 or np.sum(coords[j] == 0) == 3:
                dmap[i, j] = -1
            else:
                dmap[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
    dmap = dmap + dmap.T
    
    bins = np.concatenate(([0], np.linspace(2, 22, 31)))
    dmap = np.digitize(dmap, bins)
    dmap = (1 * (dmap < HIGH_BIN)) - (2 * (dmap == 0))

    return dmap, ''.join(secondary)


# %%
casp_contacts_sec = np.empty((13, 3), dtype='O')

for i, domain in enumerate(casp13df.index):
    realcm, realsec = get_cmap_secondary(domain)
    cp, sa = casp_eval(f'../data/our_input/contacts/{domain}.pred.rr', realcm, f'../data/our_input/secondary_structures/{domain}.pred.ss', realsec)
    
    casp_contacts_sec[i] = [domain, cp, sa]

# %%
casp_cs_df = pd.DataFrame(casp_contacts_sec[:, 1:], casp_contacts_sec[:, 0], columns=['Contact_Precision', 'Secondary_Accuracy'] )
casp_cs_df.index.name = 'Target'

casp_cs_df.to_csv('../steps/casp_contacts_secondary.csv')
# %%
casp_cs_df = pd.read_csv('../steps/casp_contacts_secondary.csv', index_col=0)

# %%
aa = casp13df.reset_index()
df = aa.melt('Target', casp13df.columns, 'Team', 'TM-Score')

bb = casp_cs_df.reset_index()
df2 = bb.melt('Target', casp_cs_df.columns)

fig = plt.figure(figsize=(12, 7))

sns.barplot('Target', 'TM-Score', hue='Team', data=df)
sns.stripplot('Target', 'value', hue='variable', data=df2, size=10, palette=['red', 'navy'])

plt.axhline(0.5, ls='--', c='green', alpha=0.5)
#plt.axhline(0.3, ls='--', c='yellow', alpha=0.5)
plt.axhline(0.17, ls='--', c='red', alpha=0.5)

plt.legend(loc = 'lower left')

plt.xticks(rotation=45)

plt.xlabel('')
plt.ylabel('TM-Score/Con. Precision/Sec. Accuracy')
plt.tight_layout()

#plt.savefig('../plots/casp13_tm_contacts_sec.png')


# %% LX contacts with |i - j| > D

def lcontacts(domain, domain_path, D=23, real_cm=None):
    
    # load prediction
    dg, sec, phi, psi = torch.load(domain_path)
    
    # load label
    if real_cm is None:
        real_cm = torch.from_numpy(contact_map(f'../data/our_input/contacts/{domain}.pred.rr'))
    
    # dg to contact probabilitites
    dg = 0.5 * (dg + dg.permute(0, 1, 3, 2)) 
    cp = torch.sum(dg[0, 1:12, :, :], dim=0)

    L = len(cp)
    
    # extract probabilities and contacts for residues at least "D" away
    cells = int((L - D) * (L - D - 1) / 2)
    contact_list = torch.empty((cells, 2))
    ind = 0
    for i in range(L-1):
        for j in range(i+1, L):
            if np.abs(i - j) > D:
                contact_list[ind] = torch.tensor([cp[i, j], real_cm[i, j]])
                ind += 1
    
    
    contact_list = contact_list[contact_list[:, 0] > 0.5]
    if len(contact_list) == 0:
        return domain, np.nan, np.nan, np.nan, np.nan
    
    precisions = [domain]
    
    for X in [5, 2, 1]:
        if len(contact_list) < (L // X):
            precisions.append(np.nan)
        else:
            L_contacts = contact_list[torch.argsort(contact_list[:, 0])][-(L//X):]
            precisions.append((torch.sum(L_contacts[:, 1]) / len(L_contacts)).item())
        
    # Full length contacts
    precisions.append(torch.mean(contact_list[:, 1]).item())
    return precisions

# %% Long range (|i-j| > 23) L/5 contacts for test domains
test_contacts = np.empty((500, 5), dtype='O')

for i, domain in enumerate(test_domains):
    test_contacts[i] = lcontacts(domain, f'../steps/test_predictions/{domain}.pred.pt', D = 23)
    
    if i % 10 == 0:
        print(f'{i} domains done')
        
# %%
lcontacts_df = pd.DataFrame(test_contacts[:, 1:], test_contacts[:, 0], columns = ['L5', 'L2', 'L', 'FL'])
lcontacts_df.index.name = 'Domain'

lcontacts_df.to_csv('../steps/long_contacts_test.csv')

# %% casp_domains
casp_contacts = np.empty((13, 5), dtype='O')
for i, domain in enumerate(casp13df.index):
    cmap, sec = get_cmap_secondary(domain)
    casp_contacts[i] = lcontacts(domain, f'../steps/casp13_predictions/{domain}.pred.pt', D = 23, real_cm = cmap)

# %%
caspcontacts_df = pd.DataFrame(casp_contacts[:, 1:], casp_contacts[:, 0], columns = ['L5', 'L2', 'L', 'FL'])
caspcontacts_df.index.name = 'Target'

caspcontacts_df.to_csv('../steps/long_contacts_casp.csv')

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

HISTCOL = 'C0'
EDGE = 'black'
MEANCOL = 'black'
RANDCOL = 'C1'
BINS = 32
ALPHA = 0.8

ax[0].hist(lcontacts_df.FL, density=True, bins=BINS, color=HISTCOL, edgecolor=EDGE, alpha=ALPHA)
ax[0].axvline(np.mean(lcontacts_df.FL), ls='--', c=MEANCOL, alpha=0.5)
ax[0].text(np.mean(lcontacts_df.FL) - 0.122 * (1.05 - 0), 0.9 * 15.5, f"{np.round(np.mean(lcontacts_df.FL), 3)}")
ax[0].text(np.mean(lcontacts_df.FL) - 0.15 * (1.05 - 0), 0.85 * 15.5, r"$\pm$" + f"{np.round(np.std(lcontacts_df.FL), 3)}")
ax[0].set_xlim(0, 1.05)
ax[0].set_ylim(0, 15.5)
ax[0].set_xlabel('Long Range (|i - j| > 23) Contact Precision')
ax[0].set_ylabel('Density')
ax[0].axvline(0.5, ls='--', c=RANDCOL)

ax[1].hist(df.loc[:, 'Secondary_Accuracy'], density=True, bins=BINS, color=HISTCOL, edgecolor=EDGE, alpha=ALPHA)
ax[1].axvline(np.mean(df.loc[:, 'Secondary_Accuracy']), ls='--', c=MEANCOL, alpha=0.5)
ax[1].text(np.mean(df.loc[:, 'Secondary_Accuracy']) - 0.122 * (1.05 - 0.25), 0.9*6.5, f"{np.round(np.mean(df.loc[:, 'Secondary_Accuracy']), 3)}")
ax[1].text(np.mean(df.loc[:, 'Secondary_Accuracy']) - 0.15 * (1.05 - 0.25), 0.85*6.5, r'$\pm$' + f"{np.round(np.std(df.loc[:, 'Secondary_Accuracy']), 3)}")
ax[1].set_xlim(0.25, 1.05)
ax[1].set_ylim(0, 6.5)
ax[1].set_xlabel('Secondary Structure Accuracy (Q3)')
ax[1].set_ylabel('Density')
ax[1].axvline(0.33, ls='--', c=RANDCOL)
plt.tight_layout()

plt.savefig('../plots/long_range_contacts_secondary_eval.png')

# %%
caspcontacts_df.rename(columns = {'FL':'FL_Precision'}, inplace = True)
# %%
aa = casp13df.reset_index()
df = aa.melt('Target', casp13df.columns, 'Team', 'TM-Score')

fig = plt.figure(figsize=(12, 7))

sns.barplot('Target', 'TM-Score', hue='Team', data=df)


lr_con_sec = pd.merge(caspcontacts_df.FL_Precision, casp_cs_df.Secondary_Accuracy, left_index=True, right_index=True).reset_index().melt('Target')
sns.stripplot('Target', 'value', hue='variable', data=lr_con_sec.reset_index(), size=10, palette=['red', 'navy'])
#sns.stripplot('Target', 'FL', data=caspcontacts_df.reset_index(), size=10, palette=['navy'])
#sns.stripplot('Target', 'Secondary_Accuracy', data=casp_cs_df.reset_index(), size=10, palette=['red'])


plt.axhline(0.5, ls='--', c='green', alpha=0.5)
#plt.axhline(0.3, ls='--', c='yellow', alpha=0.5)
plt.axhline(0.17, ls='--', c='red', alpha=0.5)

plt.legend(loc = 'lower left')

plt.xticks(rotation=45)

plt.xlabel('')
plt.ylabel('TM-Score/Con. Precision/Sec. Accuracy')
plt.tight_layout()

plt.savefig('../plots/long_range_casp13_tm_contacts_sec.png')

# %% 8 CLASS SECONDARY STRUCTURE

def ss_8(target):
    p = torch.load(f'../steps/test_predictions/{target}.pred.pt')
    y = torch.load(f'../data/our_input/Y_tensors/{target}_Y.pt')
         
    psec = torch.argmax(p[1][0], dim = 0)
    ysec = y[1].to(torch.long) - 1
    
    acc = torch.sum(psec == ysec, dtype=torch.float) / len(ysec)
    return acc.item()

# %%
test_8ss = np.empty((500, 2), dtype='O')

for i, d in enumerate(test_domains):
    test_8ss[i] = [d, ss_8(d)]
    
    if i % 10 == 0:
        print(f'{i} domains done')

# %%
ss8df = pd.DataFrame(test_8ss[:, 1], test_8ss[:, 0], columns = ['DSSP_Accuracy'])
ss8df.index.name = 'Domain'

ss8df.to_csv(f'../steps/secondary_8class_accuracy.csv')

# %%
print(np.mean(test_8ss[:, 1])) # = 0.728
print(np.std(test_8ss[:, 1])) # = 0.12