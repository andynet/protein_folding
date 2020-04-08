#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate distance maps (32 bin and 64 bin) and sequence and save them in
directories "data/our_data/distance_maps/distance_maps[32,64]" and
            "data/our_data/distance_maps/sequences"
"""

# %% Imports
from outputs_and_sequences import outputs_seq, domains
import torch
import numpy as np


# %% Function for generating .fasta file
def make_fasta(domain, sequence):
    with open(f'../../data/our_input/sequences/{domain}.fasta', 'w') as s:
        s.write(f'>{domain}\n{sequence}')


# %% Generate sequences and distance maps: DONE

# for i in range(len(domains)):
#     domain = list(domains.keys())[i]

#     d64, d32, seq, sectorsions = outputs_seq(domain)

#     if d64 is None:
#         pass
#     else:
#         make_fasta(domain, seq)
#         torch.save(torch.from_numpy(d64), f'../../data/our_input/distance_maps/distance_maps64/{domain}.pt')
#         torch.save(torch.from_numpy(d32), f'../../data/our_input/distance_maps/distance_maps32/{domain}.pt')
#         print(f'Iteration {i}, domain {domain} files generated')

# %% 19705, domain = 3zeuB02 PDBException: Structure/DSSP mismatch at <Residue MET het=  resseq=194 icode= >
for i in range(13022, len(domains)):
    domain = list(domains.keys())[i]

    d64, d32, seq, sectorsions = outputs_seq(domain, virtualcb=True)

    if d64 is None:
        pass
    else:
        torch.save(torch.from_numpy(d64), f'../../data/our_input/distance_maps/distance_maps64_cb/{domain}.pt')
        torch.save(torch.from_numpy(d32), f'../../data/our_input/distance_maps/distance_maps32_cb/{domain}.pt')
        with open(f'../../data/our_input/secondary/{domain}.sec', 'w') as f:
            for row in sectorsions:
                f.write(f'{row[0]}')
        torch.save(torch.from_numpy(sectorsions[:, 1].astype(np.int)), f'../../data/our_input/torsion/phi/{domain}_phi.pt')
        torch.save(torch.from_numpy(sectorsions[:, 2].astype(np.int)), f'../../data/our_input/torsion/psi/{domain}_psi.pt')
        print(f'Iteration {i}, domain {domain} files generated')
