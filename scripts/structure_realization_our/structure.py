#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein Geometry Class
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Bio.PDB.Polypeptide import one_to_three
from datetime import date
from mpl_toolkits.mplot3d import Axes3D
import pickle
from distributions import *

# Angles
CNCA = torch.tensor(np.radians(122))
NCAC = torch.tensor(np.radians(111))
CACN = torch.tensor(np.radians(116))
CCACB = torch.tensor(np.radians(150)) # this is the angle between the plane
    # where backbone atoms are and vector CACB (counter-closkwise)

# distances
CAC = 1.52
CN = 1.33
NCA = 1.45
CACB = 1.52


class Geometry_tools:
    def __init__(self):
        return
    
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
        
        My Implementation - commented is official rodrigues formula
        
        Input:
            v    : a 1D torch tensor
            k    : a 1D unit torch tensor
            angle: an angle in radians as a torch tensor
        
        Output:
            rotated vector: 1D torch tensor
        """

        #cp = self.cross_product(k, v)

        #vrot = v * torch.cos(angle) + cp * torch.sin(angle) + k * (torch.sum(k * v) * (1 - torch.cos(angle)))
        
        # redefine axis system to 3 new axes with unit vectors n, k and m
        n = self.cross_product(k, v)
        n = n / torch.sqrt(torch.sum(n ** 2))

        m = self.cross_product(n, k)
        m = m / torch.sqrt(torch.sum(m ** 2))

        kv = torch.sum(k * v)
        mv = torch.sum(m * v)

        v_s = torch.sqrt(torch.sum(v ** 2))

        k_axis = k * kv
        n_axis = n * torch.sin(angle) * mv
        m_axis = m * torch.cos(angle) * mv

        vrot = k_axis + n_axis + m_axis
        
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
    def __init__(self, domain, random_state=1618, kappa=8):
        self.domain = domain
        self.random_state = random_state
        
        # Load Predictions
        with open(f'../../steps/predicted_outputs/{domain}.out', 'rb') as f:
            d = pickle.load(f)

        self.distogram, phi, psi = d['distogram'], d['phi'], d['psi']
        
        # remove first phi angle and last psi angle
        # necessary because the first atom we place is Nitrogen and last is Carbon-C
        phi = phi[:, :, :, 1:]
        psi = psi[:, :, :, :-1]

        # sample angles from von Mises distribution fitted to each histogram in angleograms
        phi_sample, psi_sample = sample_torsion(phi, psi, kappa_scalar=kappa, random_state=random_state)

        # fit continuous von Mises distribution to each histogram in angleograms
        self.vmphi = fit_vm(phi)
        self.vmpsi = fit_vm(psi)

        with open(f'../../data/our_input/sequences/{domain}.fasta') as f:
            f.readline()  # fasta header
            seq = f.readline()
        
        self.torsion = torch.cat((phi_sample, psi_sample)).requires_grad_()
        self.seq = seq
        
        self.opt = torch.optim.LBFGS([self.torsion])
        
        #if len(self.phi) != len(self.psi):
        #    return 'torsion angle lengths do not match'
        
        #if len(self.phi) != len(self.seq) - 1:
        #    return 'length of torsion angles has to be one less than the sequence'
        
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
        
        phi, psi = self.torsion[:len(self.torsion) // 2], self.torsion[len(self.torsion) // 2:]
        
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
            #angles = [self.psi[i-1], torch.tensor(np.pi), self.phi[i-1]]
            angles = [psi[i-1], torch.tensor(np.pi), phi[i-1]]
            
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

        return dist_map + dist_map.t()
    
    def copy(self):
        return copy(self)
    
    def NLLLoss(self, distogram, vmphi, vmpsi):
        """
        Loss Function consisting of two potentials:
            distance potential
            torsion angle potential

        distance potential is the log of a probability of a distance value
        from a distribution to which a cubic spline is fitted

        angle potential 
        """

        x = torch.linspace(2, 22, 31)
        xtorsion = torch.linspace(-np.pi, np.pi, 36)

        loss = 0
        # DISTANCE POTENTIAL
        distance_map = self.G()
        for i in range(len(distance_map)):
            for j in range(len(distance_map)):
                if i == j:
                    pass
                else:
                    loss += 1/2 * torch.log(max(torch.tensor(0.001),
                                                interp(x, distogram[1:, i, j], min(torch.tensor(22), 
                                                                                   distance_map[i, j]))))
        # TORSION ANGLE POTENTIAL
        for i in range(len(self.torsion) // 2):
            # torsion angle loss phi
            loss += vmphi[i].log_prob(self.torsion[i])
            # torsion angle loss psi
            loss += vmpsi[i].log_prob(self.torsion[len(self.torsion) // 2 + i])

        return -loss
    
    def optimize(self, iterations, output_dir='', verbose=1):
        """
        L-BFGS Structure optimization
        
        Input:
            iterations: int, Number of iterations
            output_dir: path where a dictionary with best structure, its loss and history should be saved
            verbose   : controls the frequency of printing.  
        """
        
        
        min_loss = np.inf
        history = []
        for i in range(iterations):
            
            def closure():
                self.opt.zero_grad()
                L = self.NLLLoss(self.distogram, self.vmphi, self.vmpsi)
                L.backward()
                return L
            
            with torch.no_grad():
                loss = self.NLLLoss(self.distogram, self.vmphi, self.vmpsi)
            
            if verbose > 0:
                if i % verbose == 0:
                    print('Iteration: {:03d}, Loss: {:7.3f}'.format(i, loss.item()))
                
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_structure = self.copy()
                
            history.append([i, loss.item()])
            #start = time.time()
            self.opt.step(closure)
            #print(time.time() - start)
            
        if output_dir != '':
            o = {'structure':best_structure, 'loss':min_loss, 'history':history}
            with open(f'{output_dir}/{self.domain}_{self.random_state}.pkl', 'wb') as f:
                pickle.dump(o, f)
        else:
            return best_structure, min_loss, history
    

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
            phi, psi = self.torsion[:len(self.torsion)//2], self.torsion[len(self.torsion)//2:]
            # Initialize coordinates <=> place first 3 atoms in the space
            backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                           [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                           [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                          ])

            for i in range(len(phi)):
                atoms = ['N', 'CA', 'C']
                #angles = [self.psi[i], torch.tensor(np.pi), self.phi[i]]
                angles = [psi[i], torch.tensor(np.pi), phi[i]]
                
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
        with torch.no_grad():
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
    
    
    def pdb_atom(self, ind, a, aa, chain, pos, xyz):
        """
        PDB file ATOM template
        Input:
            ind  : int, atom index
            a    : str, atom ('N', 'CA', 'C' or 'CB')
            aa   : char, one letter aminoacid name
            chain: char, chain id character
            pos  : aminoacid position
            xyz  : list of coordinates
        
        Output:
            atom: pdb like ATOM list
        """
        atom = 'ATOM {:>6}  {:3} {:3} {:1} {:>4}   '.format(ind, a, one_to_three(aa), chain, pos)
        atom += '{:7.3f} {:7.3f} {:7.3f} X X X'.format(xyz[0], xyz[1], xyz[2])
        return atom

    def pdb_coords(self, domain_start=0, output_dir=None):
        """
        Coordinates in PDB format
        
        Input:
            self        : structure
            domain_start: index of domain start position
            output_dir  : path to a directory where pdb file should be stored
        Output:
            list of pdb_atom lists
        """
        backbone, cbeta = self.G_full()
        seq = self.seq
        
        chain = self.domain[4]
        # Round
        backbone = np.round(backbone.data.numpy(), 4)
        cbeta = np.round(cbeta.data.numpy(), 4)

        coords_full = []

        ind = 0
        bind = 0 # backbone ind
        cbind = 0
        for i in range(len(seq)):

            # Backbone atoms
            for j, a in enumerate(['N', 'CA', 'C']):
                coords_full.append(self.pdb_atom(ind + j, a, seq[i], chain, domain_start + i, backbone[bind+j]))

            ind += 3

            # C beta atom
            if seq[i] != 'G':
                coords_full.append(self.pdb_atom(ind, 'CB', seq[i], chain, domain_start + i, cbeta[cbind]))
                cbind += 1
                ind += 1

            bind += 3
        
        if output_dir is not None:
            with open(f'{output_dir}/{self.domain}_pred.pdb', 'w') as f:
                f.write('HEADER ' + str(date.today()) + '\n')
                f.write(f'TITLE Prediction of {self.domain}\n')
                f.write('AUTHOR Thinking Potato\n')
                for i in range(len(coords_full)):
                    f.write(coords_full[i] + '\n')
        else:
            return coords_full
        
