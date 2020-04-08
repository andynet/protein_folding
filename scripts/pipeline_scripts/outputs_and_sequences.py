#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating distance maps (32 bin and 64 bin), sequences, secondary
structures and torsion angles
"""
# %% Imports
from Bio.PDB import PDBParser, protein_letters_3to1
from Bio.PDB.vectors import rotaxis
import numpy as np
from secondary_torsions import secondary_torsions


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
with open(f'../../data/our_input/cath-domain-seqs-S35.fa') as f:
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


# %% Virtual C-beta atom on Glycine residue
def virtual_cbeta(residue):
    """Copied from: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ

    Function Calculates Atomic coordinates of virtual C-beta atom for glycine residue
    """

    # get atom coordinates as vectors
    n = residue['N'].get_vector()
    c = residue['C'].get_vector()
    ca = residue['CA'].get_vector()
    # center at origin
    n = n - ca
    c = c - ca
    # find rotation matrix that rotates n -120 degrees along the ca-c vector
    rot = rotaxis(-np.pi * 120.0 / 180.0, c)
    # apply rotation to ca-n vector
    cb_at_origin = n.left_multiply(rot)
    # put on top of ca atom
    cb = cb_at_origin + ca

    return [round(i, 3) for i in cb]


# %%
def get_output(domain, virtualcb=False):
    """
    Function loads pdb file, reads it and returns the atomic coordinates of
    domain with ranges specified in CATH sequence file. It also returns a list
    with secondary structure representation developed by DSSP, as well as torsion angles.

    Function can create a virtual C-beta atom of Glycine residue if requested
    Input:
        domain   : Full domain name (eg. 16pkA01)
        virtualcb: (Boolean) Create virtual C beta atom on Glycine. By default
                   is False, which means that atomic coordinates for Glycine are C-alpha

    Output:
        coords_list : 2D array of coordinates
            legend: [residue number, residue name, X, Y, Z]

        sectorsions : 2D array
            legend: [Secondary structure, Phi, Psi]
    """

    domain_start, domain_end = domains[domain][0], domains[domain][1]
    domain_id = domain[:4]
    chain_id = domain[4]

    # Get PDB structure
    try:
        structure = PDBParser().get_structure('', f'../../data/pdbfiles/{domain_id}.pdb')
    except (IndexError, ValueError):
        return None, None

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

        if residue_oneletter == 'G':  # if Glycin -> C-alpha/Virtual C-beta. Otherwise C-beta
            try:
                if virtualcb:
                    atom = virtual_cbeta(residue)
                else:
                    atom = residue['CA']
            except KeyError:
                if residue_number < domain_start:
                    if virtualcb:
                        atom = [0, 0, 0]
                    else:
                        atom = residue.child_list[0]  # just append any atom, it doesnt matter
                else:
                    print('Missing C-alpha atom')
                    return None, None
        else:
            try:
                atom = residue['CB']
            except KeyError:
                if residue_number < domain_start:
                    atom = residue.child_list[0]  # just append any atom, it doesnt matter
                else:
                    print('Missing C-beta atom')
                    return None, None

        if residue_oneletter == 'G' and virtualcb:
            x, y, z = atom
        else:
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
        return None, None

    if (end - start) == (domain_end - domain_start):
        coords_list = coords_list[start:(end + 1)]

        # Secondary structure and Torsion Angles
        sec_torsions, seq = secondary_torsions(domain, start, end)

        # Sanity check
        if seq == ''.join(coords_list[:, 2]):
            return coords_list, sec_torsions
        else:
            print('DSSP Sequence != PDB sequence')
            print(f'PDB Sequence:\n{"".join(coords_list[:, 2])}\nDSSP sequence:\n{seq}')
            return None, None
    else:
        print(f'Domain {domain} has missing data. PDB indices:{start, end}, CATH indices: {domain_start, domain_end}')
        return None, None


# %%
def outputs_seq(domain, virtualcb=False):
    """ Wrapper Function
    Generates distance matrix with 64 and 32 bins as well and outputs them
    together with sequence, secondary structure and torsion angles

    Function can create a virtual C-beta atom of Glycine residue if requested
    Input:
        domain   : Full domain name (eg. 16pkA01)
        virtualcb: (Boolean) Create virtual C beta atom on Glycine. By default
                   is False, which means that atomic coordinates for Glycine are C-alpha

    Output:
        coords_list : 2D array of coordinates
            legend: [residue number, residue name, X, Y, Z]

        sectorsions : 2D array
            legend: [Secondary structure, Phi, Psi]
    """

    coords, sectorsions = get_output(domain, virtualcb)

    if coords is None:
        return None, None, None, None

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
    return dist_mat64, dist_mat32, ''.join(sequence), sectorsions
