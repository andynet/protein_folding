#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating distance maps (32 bin and 64 bin) and sequence and saves the in a
directory "data/our_data/distance_maps/distance_maps[32,64]" and
          "data/our_data/distance_maps/sequences"
"""
# %% Imports
from Bio.PDB import PDBParser, protein_letters_3to1
import numpy as np

path = '/faststorage/project/deeply_thinking_potato'


# %%
def str_to_int(s):
    nums = '0123456789'
    for i in s:
        if i in nums:
            pass
        else:
            return None
    return int(s)


domains = {}
with open(f'{path}/data/our_input/cath-domain-seqs-S35.fa') as f:
    while True:
        info = f.readline()
        if info == '':
            break

            # extract domain id and positions
        domain, pos = info.strip().split('|')[2].split('/')
        pos = pos.split('_')  # in case the domain is not continuous

        if len(pos) == 1:  # we only allow continous domains
            start, end = pos[0].strip('-').split('-')
            start, end = str_to_int(start), str_to_int(end)  # check if there is a character in position
            if start is None or end is None:
                f.readline()
            else:
                domains[domain] = [
                    start, end,
                    f.readline().strip()]
        else:
            f.readline()

del info, start, end, f, domain, pos


# %%
def get_coords(domain):

    domain_start, domain_end = domains[domain][0], domains[domain][1]
    domain_id = domain[:4]
    chain_id = domain[4]

    # Get PDB structure
    try:
        structure = PDBParser().get_structure('', f'{path}/data/pdbfiles/{domain_id}.pdb')
    except (IndexError, ValueError):
        return

    # There is a problem with 0 character, because sometimes
    # it means no chain (chain == ' '), but another times
    # it is a valid chain ID
    if chain_id == '0':
        # get all chain_IDs
        chain_IDs = np.array([ch.get_full_id()[2] for ch in structure.get_chains()])
        if '0' in chain_IDs:
            pass
        else:
            chain_id = ' '

    chain = structure[0][chain_id]

    coords_list = []

    known_aminoacids = np.array(list(protein_letters_3to1.keys()))

    for i, residue in enumerate(chain.get_residues()):
        residue_name = residue.get_resname()

        if residue_name not in known_aminoacids:
            break

        residue_oneletter = protein_letters_3to1[residue_name]
        residue_number = residue.child_list[0].get_full_id()[3][1]

        if residue_oneletter == 'G':  # if Glycin -> C-alpha. Otherwise C-beta
            try:
                atom = residue['CA']
            except KeyError:
                if residue_number < domain_start:
                    atom = residue.child_list[0]  # just append any atom, it doesnt matter
                else:
                    print('Missing C-alpha atom')
                    return
        else:
            try:
                atom = residue['CB']
            except KeyError:
                if residue_number < domain_start:
                    atom = residue.child_list[0]  # just append any atom, it doesnt matter
                else:
                    print('Missing C-beta atom')
                    return

        x, y, z = atom.get_coord()
        coords_list.append([residue_number,
                            residue_name,
                            residue_oneletter,
                            x, y, z])

        if residue_number == domain_end:  # because we need to include also that residue
            break

    coords_list = np.array(coords_list, dtype='O')

    # in case the domain_start is not included in the coords indices
    try:
        start = np.where(coords_list[:, 0] == domain_start)[0][0]
        end = np.where(coords_list[:, 0] == domain_end)[0][0]
    except IndexError:
        print('domain_start or domain_end index not found in pdb file')
        return

    if (end - start) == (domain_end - domain_start):
        return coords_list[start:(end + 1)]
    else:
        print(f'Domain {domain} has missing data. PDB indices:{start, end}, CATH indices: {domain_start, domain_end}')
        return


# %%
def dist_mat_seq(domain):
    """ Generates distance matrix with 64 and 32 bins as well as the sequence"""

    coords = get_coords(domain)

    if coords is None:
        return None, None, None

    L = coords.shape[0]

    dist_mat = np.zeros((L, L))

    for i in range(L - 1):
        for j in range(i + 1, L):
            dist_mat[i, j] = np.sqrt(np.sum((coords[i, 3:] - coords[j, 3:]) ** 2))

    bins64 = np.concatenate(([0], np.linspace(2, 22, 62), [1000]))
    bins32 = np.concatenate(([0], np.linspace(2, 22, 30), [1000]))

    dist_mat += dist_mat.T

    dist_mat64 = np.digitize(dist_mat, bins64)
    dist_mat32 = np.digitize(dist_mat, bins32)

    sequence = coords[:, 2]
    return dist_mat64, dist_mat32, ''.join(sequence)
