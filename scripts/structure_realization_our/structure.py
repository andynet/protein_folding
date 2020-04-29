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

class Structure:
    def __init__(self, phi, psi, Y, seq):
        self.phi = phi
        self.psi = psi
        self.Y = Y
        self.seq = seq
    
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
            angle = CACN
        elif atom == 'CA':
            v_size = NCA
            angle = CNCA
        elif atom == 'C':
            v_size = CAC
            angle = NCAC

        k = coords[-1] - coords[-2]
        k = k / torch.sqrt(torch.sum(k ** 2))

        n = self.cross_product(coords[-3] - coords[-2], coords[-1] - coords[-2])
        n = n / torch.sqrt(torch.sum(n ** 2))

        v = torch.cos(np.pi - angle) * k + \
            torch.sin(np.pi - angle) * self.cross_product(n, k)
        return v * v_size
    
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

        v = torch.cos(CCACB) * k + \
                torch.sin(CCACB) * self.cross_product(n, k)

        return v * CACB + residue[1]
    
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
            angles = [np.pi - self.psi[i - 1], np.pi - torch.tensor(np.pi), np.pi - self.phi[i - 1]]

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

        # Initialize coordinates <=> place first 3 atoms in the space
        backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                       [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                       [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                      ])

        for i in range(len(self.phi)):
            atoms = ['N', 'CA', 'C']
            angles = [np.pi - self.psi[i], np.pi - torch.tensor(np.pi), np.pi - self.phi[i]]

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
    
    def visualize_structure(self):
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
        
################## THIS DOESNT WORK!!! #######################
    def calc_prob(self, x, _mean, _sd):
        """
        Calculates the output of a probability function at value x. The probability function
        is normal with center at _mean and standard deviation _sd
        """

        return torch.exp(- (1 / 2) * ((x - _mean) / _sd) ** 2) / (_sd * torch.sqrt(torch.tensor(2 * np.pi)))
    
    def loss(self, pred, sd = torch.tensor(4)):
        """NLL loss"""
        
        loss = 0

        for i in range(len(pred) - 1):
            for j in range(i + 1, len(pred)):
                prob = self.calc_prob(pred[i, j], self.Y[i, j], sd)
                loss += torch.log(prob)
        return -loss
    
    def optimize(self, iterations=10, lr=1e-5, lr_decay = 1, verbose=1):
    
        history = []
        for i in range(iterations):
            if self.phi.grad is not None:
                self.phi.grad.zero_()
            if self.psi.grad is not None:
                self.psi.grad.zero_()

            temp_distmap = self.G()
            L = self.loss(temp_distmap, self.Y)
            L.backward()
            self.phi = (self.phi - lr * self.phi.grad).detach().requires_grad_()
            self.psi = (self.psi - lr * self.psi.grad).detach().requires_grad_()

            if verbose is not None:
                if i % verbose == 0:
                    print(f'Iteration {i}, Loss: {L.item()}')
            history.append([i, L.item()])
            lr /= lr_decay
        return np.array(history)