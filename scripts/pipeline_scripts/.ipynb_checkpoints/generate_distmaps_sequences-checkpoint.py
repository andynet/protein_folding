#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate distance maps (32 bin and 64 bin) and sequence and save them in
directories "data/our_data/distance_maps/distance_maps[32,64]" and
            "data/our_data/distance_maps/sequences"
"""

# %% Imports
import os
path = '/faststorage/project/deeply_thinking_potato'
os.chdir(f'{path}/scripts/pipeline_scripts/')

from distance_maps_and_sequences import dist_mat_seq, domains
import torch


# %% Function for generating .fasta file
def make_fasta(domain, sequence):
    with open(f'{path}/data/our_input/sequences/{domain}.fasta', 'w') as s:
        s.write(f'>{domain}\n{sequence}')


# %% Generate sequences and distance maps: DONE
for i in range(len(domains)):
    domain = list(domains.keys())[i]

    d64, d32, seq = dist_mat_seq(domain)

    if d64 is None:
        pass
    else:
        make_fasta(domain, seq)
        torch.save(torch.from_numpy(d64), f'{path}/data/our_input/distance_maps/distance_maps64/{domain}.pt')
        torch.save(torch.from_numpy(d32), f'{path}/data/our_input/distance_maps/distance_maps32/{domain}.pt')

        print(f'Iteration {i}, domain {domain} files generated')
