#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:56:04 2020

@author: tomasla

For more explanation is the geometry.ipynb
"""

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %% Constants - see atomic distances.ipynb

# Angles
CNCA = torch.tensor(np.radians(122))
NCAC = torch.tensor(np.radians(111))  # ??? i am not sure about this one - From atomic_coordinates.ipynb it should be ~111
CACN = torch.tensor(np.radians(116))
CCACB = torch.tensor(np.radians(120))

# distances
CAC = 1.52
CN = 1.33
NCA = 1.45
CACB = 1.52
CACB = 1.52

# minimal distance between two same atoms (C-C, N-N, CA-CA)
CC = CN * torch.cos(np.pi - CNCA) + NCA + CAC * torch.cos(np.pi - NCAC)
NN = NCA * torch.cos(np.pi - NCAC) + CAC + CN * torch.cos(np.pi - CACN)
CACA = CAC * torch.cos(np.pi - CACN) + CN + NCA * torch.cos(np.pi - CNCA)

# %%
def cross_product(k, v):

    # definition of cross product
    cp = torch.tensor([
        k[1] * v[2] - k[2] * v[1],
        k[2] * v[0] - k[0] * v[2],
        k[0] * v[1] - k[1] * v[0]
    ])
    return cp

# %%
def calc_v(coords, atom):

    if atom == 'N':
        v_size = CN
        angle = CACN
    elif atom == 'CA':
        v_size = NCA
        angle = CNCA
    elif atom == 'C':
        v_size = CAC
        angle = NCAC

    k = coords[-1] - coords[-2]
    k = k / torch.sqrt(torch.sum(k ** 2))

    n = cross_product(coords[-3] - coords[-2], coords[-1] - coords[-2])
    n = n / torch.sqrt(torch.sum(n ** 2))

    v = torch.cos(np.pi - angle) * k + \
        torch.sin(np.pi - angle) * cross_product(n, k)
    return v * v_size


# %% Rodrigues Formula
def rodrigues(v, k, angle):
    """
    see: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    
    cp = cross_product(k, v)
    
    vrot = v * torch.cos(angle) + cp * torch.sin(angle) + k * (torch.sum(k * v) * (1 - torch.cos(angle)))
    return vrot


# %%
def calc_atom_coords(coord_list, atom, angle):
    k = coord_list[-1] - coord_list[-2]
    k = k / torch.sqrt(torch.sum(k ** 2))  # unit vector
    
    v = calc_v(coord_list, atom)

    rotated = rodrigues(v, k, angle)
    return rotated + coord_list[-1]


# %%
def calc_dihedral(coords):
    """see https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates"""
    
    if len(coords) != 4:
        return 'Input should be a list of 4 atoms'
    a1, a2, a3, a4 = coords
    b1 = a2 - a1
    b2 = a3 - a2
    b3 = a4 - a3
    
    b2_n = b2 / torch.sqrt(torch.sum(b2 ** 2))
    
    n1 = cross_product(b1, b2)
    n1 = n1 / torch.sqrt(torch.sum(n1 ** 2))
    
    n2 = cross_product(b2, b3)
    n2 = n2 / torch.sqrt(torch.sum(n2 ** 2))
    
    x = torch.sum(n1 * n2)
    m = cross_product(n1, b2_n)
    y = torch.sum(m * n2)
    
    return -torch.atan2(y, x)


# %%
def place_cbeta(residue):
    """
    Calculate coordinates of C-beta atom. Input is a list of coordinates 
    of 3 atoms in the order: N, CA, C
    
    Returns coordinates of the C-beta atom
    """
    
    v1 = residue[0] - residue[1]  # vector CA-N
    v2 = residue[2] - residue[1]  # vector CA-C

    v1_scaled = CAC * (v1 / torch.sqrt(torch.sum(v1 ** 2)))

    n = v2 - v1_scaled
    n = n / torch.sqrt(torch.sum(n ** 2))

    k = cross_product(v2, v1)
    k = k / torch.sqrt(torch.sum(k ** 2))

    v = torch.cos(CCACB) * k + \
            torch.sin(CCACB) * cross_product(n, k)

    return v * CACB + residue[1]

# %%
def G(phi, psi, sequence):
    """
    Calculate the backbone coordinates + C-beta satisfying the input torsion angles.
    The sequence has to be inputed in order to know whether a residue is glycin or not.
    Calculates coordinates only of C-beta atoms (C-alpha for glycin).
    
    Outputs distance map
    """
    
    dist_mat_atoms = torch.empty((len(sequence), 3))
    
    # Initialize coordinates <=> place first 3 atoms in the space
    backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                   [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                   [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                  ])
    
    # first c beta
    if sequence[0] == 'G':
        dist_mat_atoms[0] = backbone[1]
    else:
        dist_mat_atoms[0] = place_cbeta(backbone)
    
    i = 1
    while i < len(sequence):       
        # backbone atoms
        atoms = ['N', 'CA', 'C']
        angles = [np.pi - psi[i - 1], np.pi - torch.tensor(np.pi), np.pi - phi[i - 1]]

        for j in range(3):
            atom = calc_atom_coords(backbone, atoms[j], angles[j])
            backbone = torch.cat((backbone, atom.view(1, 3)))
        
        if sequence[i] == 'G':
            dist_mat_atoms[i] = backbone[3 * i + 1]
        else:
            dist_mat_atoms[i] = place_cbeta(backbone[(3 * i):(3 * (i + 1))])
        
        i += 1
    
    # distance_map
    dist_map = torch.zeros((len(sequence), len(sequence)))
    
    for i in range(len(sequence) - 1):
        for j in range(i + 1, len(sequence)):
            dist_map[i, j] = torch.sqrt(torch.sum((dist_mat_atoms[i] - dist_mat_atoms[j]) ** 2))
    
    return dist_map   


# %%
def G_full(phi, psi, sequence):
    """
    Calculate the backbone coordinates + C-beta satisfying the input torsion angles.
    The sequence has to be inputed in order to know whether a residue is glycin or not.
    """
    
    # Initialize coordinates <=> place first 3 atoms in the space
    backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                   [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                   [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                  ])
           
    for i in range(len(phi)):
        atoms = ['N', 'CA', 'C']
        angles = [np.pi - psi[i], np.pi - torch.tensor(np.pi), np.pi - phi[i]]

        for j in range(3):
            atom = calc_atom_coords(backbone, atoms[j], angles[j])
            backbone = torch.cat((backbone, atom.view(1, 3)))
    
    # cbeta atoms
    cbeta_coords = torch.empty((sum([i != 'G' for i in sequence]), 3))
    
    it = 0
    for i in range(len(sequence)):
        if sequence[i] != 'G':
            cbeta = place_cbeta(backbone[(i*3):(i*3+3)])
            cbeta_coords[it] = cbeta
            it += 1
            
    return backbone, cbeta_coords


# %%
def visualize_structure(backbone, cbeta_coords, seq):
    
    """Visualizes the entire structure: backbone + C-beta atoms
    
    In the first step generates a list of residue coordinates:
        If Glycin: only N, CA, C atoms are present.
        Else: N, CA, CB, CA, C (the CA is there twice to make the plotting easier)
    """
    
    fig = plt.figure()
    
    entire_structure = torch.empty((len(backbone) + 2 * len(cbeta_coords), 3))
    
    it = 0
    cb_it = 0
    for i in range(len(seq)):
        N, CA, C = backbone[(3 * i):(3 * i + 3)]
        
        if seq[i] != 'G':
            CB = cbeta_coords[cb_it]
            cb_it += 1
            entire_structure[it:(it+5)] = torch.cat([N, CA, CB, CA, C]).view((5, 3))
            it += 5
        else:
            entire_structure[it:(it+3)] = torch.cat([N, CA, C]).view((3, 3))
            it += 3      
    
    coords = entire_structure.data.numpy()

    ax = fig.gca(projection='3d')
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])

    
# %%
def make_distmap(coords):
    
    dist_map = torch.zeros((len(coords), len(coords)))
    
    for i in range(len(coords) - 1):
        for j in range(i + 1, len(coords)):
            dist_map[i, j] = torch.sqrt(torch.sum((coords[i] - coords[j]) ** 2))
    
    return dist_map + dist_map.t()