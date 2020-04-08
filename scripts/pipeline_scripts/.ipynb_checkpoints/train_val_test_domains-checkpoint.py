#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script generates lists of Training domains, Validation domains and Test domains
"""

# %% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Constants
TEST_SET_SIZE = 500
VALIDATION_SET_SIZE = 1000
HOMOLOGOUS_SUPERFAMILY_CUTOFF = 500
MAX_FAMILY_SIZE = 100

# %% Open CATH list file
with open('../../data/our_input/cath-domain-list-S35.txt') as f:
    raw = f.readlines()

# %% downloaded and generated domains and their lengths
generated_domain_lengths = {}
for i in os.listdir('../../data/our_input/sequences/'):
    with open(f'../../data/our_input/sequences/{i}') as f:
        f.readline()
        generated_domain_lengths[i.split('.')[0]] = len(f.readline())

del f
# %% Extract Domain name, Class, Architecture, Topology, Homology superfamily
# Column 1:  CATH domain name (seven characters)     0
# Column 2:  Class number                            1
# Column 3:  Architecture number                     2
# Column 4:  Topology number                         3
# Column 5:  Homologous superfamily number           4
# Column 6:  S35 sequence cluster number             5
# Column 7:  S60 sequence cluster number             6
# Column 8:  S95 sequence cluster number             7
# Column 9:  S100 sequence cluster number            8
# Column 10: S100 sequence count number              9
# Column 11: Domain length                           10
# Column 12: Structure resolution (Angstroms)        11
#            (999.000 for NMR structures and 1000.000 for obsolete PDB entries)

cath_domains = []
cath_data = np.empty((len(generated_domain_lengths), 5), dtype=np.int)
ind = 0
for i in range(len(raw)):
    line = raw[i].strip().split()
    if line[0] in list(generated_domain_lengths.keys()):
        cath_domains.append(line[0])
        cath_data[ind, 1:] = [int(line[j]) for j in [1, 2, 3, 4]]
        cath_data[ind, 0] = ind
        ind += 1

cath_domains = np.array(cath_domains)
del line, i, raw, f


# we cannot use the column 11 because the domain lengths from pdb file differ
# from the cath files (usually by one or two aminoacids)

# Generate Train/Validation/Test set

# 1. Filter Homologous Superfamilies with more than `HOMOLOGOUS_SUPERFAMILY_CUTOFF` members. 
# If this condition is satisfied then randomly pick that number of domains from that 
# particular superfamily

homologous_superfamilies, counts = np.unique(cath_data[:, 4], return_counts=True)

indices_to_delete = []

for HS_ID in homologous_superfamilies[np.where(counts > HOMOLOGOUS_SUPERFAMILY_CUTOFF)[0]]:
    HS_ind = cath_data[cath_data[:, 4] == HS_ID][:, 0]
    pick_indices_to_delete = np.random.choice(HS_ind, HS_ind.shape[0] - HOMOLOGOUS_SUPERFAMILY_CUTOFF, replace=False)
    indices_to_delete.extend(list(pick_indices_to_delete))

cath_data_filtered = cath_data[np.setdiff1d(cath_data[:, 0], indices_to_delete)]

# %% 2. Create Train, Val and Test set, while keeping in mind that all members of one 
# superfamily have to be inside on of the sets
homologous_superfamilies, counts = np.unique(cath_data_filtered[:, 4], return_counts=True)

# 2.1 Pick families
np.random.seed(1618)

length = 0
families = []

# TEST
test_families = []

while length != TEST_SET_SIZE:
    
    ind = np.random.randint(len(counts))
    family, familysize = homologous_superfamilies[ind], counts[ind]
    
    if familysize < MAX_FAMILY_SIZE:
        if length + familysize <= TEST_SET_SIZE and family not in families:
            length += familysize
            families.append(family)
            test_families.append(family)


length = 0

# VALIDATION
validation_families = []

while length != VALIDATION_SET_SIZE:
    
    ind = np.random.randint(len(counts))
    family, familysize = homologous_superfamilies[ind], counts[ind]
    
    if familysize < MAX_FAMILY_SIZE:
        if length + familysize <= VALIDATION_SET_SIZE and family not in families:
            length += familysize
            families.append(family)
            validation_families.append(family)
            
# %% 2.2 Create datasets
# TEST
test_data = cath_data_filtered[cath_data_filtered[:, 4] == test_families[0]]
for i in range(1, len(test_families)):
    test_data = np.concatenate((test_data, cath_data_filtered[cath_data_filtered[:, 4] == test_families[i]]))
    
# VALIDATION
validation_data = cath_data_filtered[cath_data_filtered[:, 4] == validation_families[0]]
for i in range(1, len(validation_families)):
    validation_data = np.concatenate((validation_data, cath_data_filtered[cath_data_filtered[:, 4] == validation_families[i]]))
    
# TRAIN
train_indices = np.setdiff1d(cath_data_filtered[:, 0], np.concatenate((test_data[:, 0], validation_data[:, 0])))
train_data = cath_data_filtered[[cath_data_filtered[i, 0] in train_indices for i in range(cath_data_filtered.shape[0])]]

# %% 2.3 Write everythin in files

# Test Domains
test_domains = cath_domains[test_data[:, 0]]
with open('../../data/our_input/test_domains.csv', 'w') as f:
    for i in test_domains:
        f.write(i + '\n')
        
# Validation Domains
validation_domains = cath_domains[validation_data[:, 0]]
with open('../../data/our_input/validation_domains.csv', 'w') as f:
    for i in validation_domains:
        f.write(i + '\n')

# Train Domains
train_domains = cath_domains[train_data[:, 0]]
with open('../../data/our_input/train_domains.csv', 'w') as f:
    for i in train_domains:
        f.write(i + '\n')