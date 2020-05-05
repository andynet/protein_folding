#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein Geometry Class
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Angles
CNCA = torch.tensor(np.radians(122))
NCAC = torch.tensor(np.radians(111))
CACN = torch.tensor(np.radians(116))
CCACB = torch.tensor(np.radians(120))

# distances
CAC = 1.52
CN = 1.33
NCA = 1.45
CACB = 1.52
CACB = 1.52


class Geometry_tools:
    def __init__(self):
        return self
       
    def cross_product(self, k, v):
        # definition of cross product
        cp = torch.tensor([
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0]
        ])
        return cp
    
    def calc_v(self, coords, atom):
        """
        Calculate vector in the plane of previous 3 atoms in the direction 
        of the target atom
        
        Input:
            coords: a 2D torch tensor of shape (L, 3)
            atom  : a string ('C', 'N' or 'CA')
        
        Output:
            vector "v": 1D torch tensor 
        """
        
        if atom == 'N':
            v_size = CN
            angle = CACN + NCAC - np.pi
        elif atom == 'CA':
            v_size = NCA
            angle = CACN + CNCA - np.pi
        elif atom == 'C':
            v_size = CAC
            angle = CNCA + NCAC - np.pi

        k = coords[-1] - coords[-2]
        k = k / torch.sqrt(torch.sum(k ** 2))

        v0 = coords[-3] - coords[-2]
        v0 = v0 / torch.sqrt(torch.sum(v0 ** 2))

        n = self.cross_product(v0, k)
        n = n / torch.sqrt(torch.sum(n ** 2))

        return v_size * self.rodrigues(v0, n, angle)
    
    def rodrigues(self, v, k, angle):
        """
        Rotate vector "v" by a angle around basis vector "k"
        see: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        
        Input:
            v    : a 1D torch tensor
            k    : a 1D unit torch tensor
            angle: an angle in radians as a torch tensor
        
        Output:
            rotated vector: 1D torch tensor
        """

        cp = self.cross_product(k, v)

        vrot = v * torch.cos(angle) + cp * torch.sin(angle) + k * (torch.sum(k * v) * (1 - torch.cos(angle)))
        return vrot
    
    def calc_atom_coords(self, coords, atom, angle):
        """
        Calculate coordinates of a new atom given list of coordinates of at least 
        previous three atoms
        
        Input:
            coords: a 2D torch tensor of shape (L, 3)
            atom  : a string ('C', 'N' or 'CA')
            angle: an angle in radians as a torch tensor
        
        Output:
            rotated and translated vector: 1D torch tensor
        """
        
        k = coords[-1] - coords[-2]
        k = k / torch.sqrt(torch.sum(k ** 2))  # unit vector

        v = self.calc_v(coords, atom)

        rotated = self.rodrigues(v, k, angle)
        return rotated + coords[-1]
    
    def place_cbeta(self, residue):
        """
        Calculate coordinates of C-beta atom. 
        
        Input:
            residue: a 2D torch tensor of coordinates in the order N, CA, C
        
        Output
            coordinates of the C-beta atom: 1D torch tensor
        """

        v1 = residue[0] - residue[1]  # vector CA-N
        v2 = residue[2] - residue[1]  # vector CA-C

        v1_scaled = CAC * (v1 / torch.sqrt(torch.sum(v1 ** 2)))

        n = v2 - v1_scaled
        n = n / torch.sqrt(torch.sum(n ** 2))

        k = self.cross_product(v2, v1)
        k = k / torch.sqrt(torch.sum(k ** 2))

        return self.rodrigues(k, n, CCACB) * CACB + residue[1]
    

class Structure(Geometry_tools):
    def __init__(self, phi, psi, Y, seq):
        self.phi = phi
        self.psi = psi
        self.Y = Y
        self.seq = seq
    
    def G(self):
        """
        Create differentiable protein geometry
        
        Input:
            phi     : 1D torch tensor
            psi     : 1D torch tensor
            sequence: string
            
        Output:
            2D tensor of coordinates
        """
        
        if len(self.phi) != len(self.psi):
            return 'torsion angle lengths do not match'
        
        if len(self.phi) != len(self.seq) - 1:
            return 'length of torsion angles has to be one less than the sequence'
        
        dist_mat_atoms = torch.empty((len(self.seq), 3))

        # Initialize coordinates <=> place first 3 atoms in the space
        backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                       [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                       [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                      ])

        # first c beta
        if self.seq[0] == 'G':
            dist_mat_atoms[0] = backbone[1]
        else:
            dist_mat_atoms[0] = self.place_cbeta(backbone)

        i = 1
        while i < len(self.seq):   
            
            # backbone atoms
            atoms = ['N', 'CA', 'C']
            angles = [self.psi[i-1], torch.tensor(np.pi), self.phi[i-1]]

            for j in range(3):
                atom = self.calc_atom_coords(backbone, atoms[j], angles[j])
                backbone = torch.cat((backbone, atom.view(1, 3)))

            if self.seq[i] == 'G':
                dist_mat_atoms[i] = backbone[3 * i + 1]
            else:
                dist_mat_atoms[i] = self.place_cbeta(backbone[(3 * i):(3 * (i + 1))])

            i += 1

        # distance_map
        dist_map = torch.zeros((len(self.seq), len(self.seq)))

        for i in range(len(self.seq) - 1):
            for j in range(i + 1, len(self.seq)):
                dist_map[i, j] = torch.sqrt(torch.sum((dist_mat_atoms[i] - dist_mat_atoms[j]) ** 2))

        return dist_map
    
    def G_full(self):
        """
        Calculate the backbone coordinates + C-beta satisfying the input torsion angles.
        The sequence has to be inputed in order to know whether a residue is glycin or not.
        
        Input:
            phi     : 1D torch tensor
            psi     : 1D torch tensor
            sequence: string
            
        Output:
            tuple
                2D tensor of Backbone coordinates
                2D tensor of C-beta coordinates
        """
        
        with torch.no_grad():
            # Initialize coordinates <=> place first 3 atoms in the space
            backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                           [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                           [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                          ])

            for i in range(len(self.phi)):
                atoms = ['N', 'CA', 'C']
                angles = [self.psi[i], torch.tensor(np.pi), self.phi[i]]
                
                for j in range(3):
                    atom = self.calc_atom_coords(backbone, atoms[j], angles[j])
                    backbone = torch.cat((backbone, atom.view(1, 3)))

            # cbeta atoms
            cbeta_coords = torch.empty((sum([i != 'G' for i in self.seq]), 3))

            it = 0
            for i in range(len(self.seq)):
                if self.seq[i] != 'G':
                    cbeta = self.place_cbeta(backbone[(i*3):(i*3+3)])
                    cbeta_coords[it] = cbeta
                    it += 1

            return backbone, cbeta_coords
    
    def visualize_structure(self, img_path=None):
        """
        Visualizes the entire structure: backbone + C-beta atoms

        In the first step generates a list of residue coordinates:
            If Glycin: only N, CA, C atoms are present.
            Else: N, CA, CB, CA, C (the CA is there twice to make the plotting easier)
        
        
        """
        backbone, cbeta_coords = self.G_full()

        fig = plt.figure()

        entire_structure = torch.empty((len(backbone) + 2 * len(cbeta_coords), 3))

        it = 0
        cb_it = 0
        for i in range(len(self.seq)):
            N, CA, C = backbone[(3 * i):(3 * i + 3)]

            if self.seq[i] != 'G':
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
        
        if img_path is not None:
            plt.savefig(img_path)
            plt.close(fig)
        