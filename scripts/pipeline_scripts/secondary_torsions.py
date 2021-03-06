#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:07:26 2020

@author: tomasla
"""
# %%
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import numpy as np


# %%
def secondary_torsions(domain):#, start, end):
    """Extract Secondary structure and torsion angles using the DSSP package"""

    domain_id = domain[:4]
    chain_id = domain[4]

    structure = PDBParser().get_structure('', f'../../data/pdbfiles/{domain_id}.pdb')
    try:
        raw = DSSP(structure[0], f'../../data/pdbfiles/{domain_id}.pdb')
    except:
        print('PDBException. Nothing we can do')
        return None, None
    dssp = np.array(raw.property_list, dtype='O')

    # extract chain
    #keys = np.array([i[0] for i in raw.keys()])
    #positions = np.array([int(i[1][1]) for i in raw.keys()])
    #positions = positions[keys == chain_id]

    #dssp = dssp[keys == chain_id]

    sequence = ''.join(dssp[:, 1])

    sec_torsions = dssp[:, [2, 4, 5]]

    # translating torsion angles to range (-180, 180)
    for i in range(sec_torsions.shape[0]):
        for j in range(1, 3):
            if sec_torsions[i, j] > 180:
                sec_torsions[i, j] = sec_torsions[i, j] - 360
            elif sec_torsions[i, j] < -180:
                sec_torsions[i, j] = 360 - sec_torsions[i, j]
    #try:
    #    dssp_start, dssp_end = np.where(positions == start)[0][0], np.where(positions == end)[0][0]
    #except IndexError:
    #    print(domain, 'positions not found')
    #    return None, None
    return sec_torsions, sequence#sec_torsions[dssp_start:(dssp_end + 1)], sequence[dssp_start:(dssp_end + 1)]


# %%
def secondary_torsions_casp(domain):#, start, end):
    """Extract Secondary structure and torsion angles using the DSSP package"""


    structure = PDBParser().get_structure('', f'../../data/pdbfiles/{domain}.pdb')
    try:
        raw = DSSP(structure[0], f'../../data/pdbfiles/{domain}.pdb')
    except:
        print('PDBException. Nothing we can do')
        return None, None
    dssp = np.array(raw.property_list, dtype='O')

    sequence = ''.join(dssp[:, 1])

    sec_torsions = dssp[:, [2, 4, 5]]

    # translating torsion angles to range (-180, 180)
    for i in range(sec_torsions.shape[0]):
        for j in range(1, 3):
            if sec_torsions[i, j] > 180:
                sec_torsions[i, j] = sec_torsions[i, j] - 360
            elif sec_torsions[i, j] < -180:
                sec_torsions[i, j] = 360 - sec_torsions[i, j]

    return sec_torsions

# %% CASP Targets
#translation = {'G': 'H', 'E': 'E', 'B': 'E', 'H': 'H', 'I': 'H', 'T': 'C', 'S': 'C', '-': 'C'}
#casp_domain_ranges = np.loadtxt('../../steps/casp_domain_ranges.csv', dtype='O', delimiter=',')

#for (domain, s, e) in casp_domain_ranges:
#    dssp = secondary_torsions_casp(domain)
#    
#    q8 = dssp[:, 0]
#    q3 = [translation[i] for i in q8]
#    with open(f'../../data/our_input/secondary/{domain}.sec', 'w') as f:
#        f.write(''.join(q8))
#    
#    with open(f'../../data/our_input/secondary_structures/{domain}.real.ss', 'w') as f:
#        f.write(''.join(q3))
    

