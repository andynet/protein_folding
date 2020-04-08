#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that generates sequences and distance maps for selected CASP 13 Regular Targets
"""

# %% Imports
from Bio import SeqIO
import requests
import torch
from casp13_data_functions import make_fasta, dist_mat

# %% Load CASP targets list
with open('../../data/casp13_test_data/casp13_targets') as f:
    raw = f.readlines()

casp_targets = {}
for i in raw:
    line = i.strip().split()
    casp_targets[line[0]] = line[1]
del raw
# %% deleted fragmented and other weird domains (because pdb)
del casp_targets['T0951']
del casp_targets['T0965']
del casp_targets['T0970']
del casp_targets['T1011']

# %% Load CASP sequences
# downloaded from http://predictioncenter.org/download_area/CASP13/sequences/casp13.seq.txt
casp_sequences = {}
for record in SeqIO.parse("../../data/casp13_test_data/casp13_seq.fa", "fasta"):
    if record.id in list(casp_targets.keys()):
        casp_sequences[record.id] = str(record.seq)


# %% 1. Download PDB files
def download_pdb(target):
    url = 'https://files.rcsb.org/download/' + target + '.pdb'
    myfile = requests.get(url)
    open(f'../../data/casp13_test_data/pdbfiles/' + target + '.pdb', 'wb').write(myfile.content)


for t in casp_targets.values():
    download_pdb(t)

# %% Generate sequences and Distance maps

for t in casp_targets:
    d64, d32 = dist_mat(t, casp_sequences, casp_targets)
    make_fasta(t, casp_sequences)
    torch.save(torch.from_numpy(d64), f'../../data/casp13_test_data/distance_maps/distance_maps64/{t}.pt')
    torch.save(torch.from_numpy(d32), f'../../data/casp13_test_data/distance_maps/distance_maps32/{t}.pt')
