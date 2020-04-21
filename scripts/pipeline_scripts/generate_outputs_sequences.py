#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate distance maps (32 bin and 64 bin) and sequence and save them in
directories "data/our_data/distance_maps/distance_maps[32,64]" and
            "data/our_data/distance_maps/sequences"
"""

# %% Imports
from outputs_and_sequences import outputs_seq
import torch
import numpy as np
import os


# %% Function for generating .fasta file
def make_fasta(domain, sequence):
    with open(f'../../data/our_input/sequences/{domain}.fasta', 'w') as s:
        s.write(f'>{domain}\n{sequence}')


# %%
domains0 = np.array([i.split('.')[0] for i in os.listdir('../../data/our_input/sequences')])
generated = np.array([i.split('.')[0] for i in os.listdir('../../data/our_input/secondary')])
not_generated = np.setdiff1d(domains0, generated)
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

# %% FOK IT. 19675 is enough
dssp_smaller_than_pdb = 0
for i in range(70, len(not_generated)):
    domain = not_generated[i]

    d64, d32, seq, sectorsions = outputs_seq(domain, virtualcb=True)
    
    if isinstance(d64, int):
        dssp_smaller_than_pdb += 1
    elif d64 is None:
        pass
        # break
    else:
        # generated_number += 1
        torch.save(torch.from_numpy(d64), f'../../data/our_input/distance_maps/distance_maps64_cb/{domain}.pt')
        torch.save(torch.from_numpy(d32), f'../../data/our_input/distance_maps/distance_maps32_cb/{domain}.pt')
        with open(f'../../data/our_input/secondary/{domain}.sec', 'w') as f:
            for row in sectorsions:
                f.write(f'{row[0]}')
        torch.save(torch.from_numpy(sectorsions[:, 1].astype(np.int)), f'../../data/our_input/torsion/phi/{domain}_phi.pt')
        torch.save(torch.from_numpy(sectorsions[:, 2].astype(np.int)), f'../../data/our_input/torsion/psi/{domain}_psi.pt')
        print(f'Iteration {i}, domain {domain} files generated')
