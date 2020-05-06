#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Optimization
"""

import torch
import numpy as np
from distributions import *
from structure import Structure
from copy import copy

def NLLLoss(structure, distogram, vmphi, vmpsi):
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
    distance_map = structure.G()
    for i in range(len(distance_map)):
        for j in range(len(distance_map)):
            if i == j:
                pass
            else:
                loss += 1/2 * torch.log(max(torch.tensor(0.001),
                                            interp(x, distogram[1:, i, j], min(torch.tensor(22), 
                                                                               distance_map[i, j]))))
    # TORSION ANGLE POTENTIAL
    for i in range(len(structure.phi)):
        # torsion angle loss phi
        loss += vmphi[i].log_prob(structure.torsion[i])
        # torsion angle loss psi
        loss += vmpsi[i].log_prob(structure.torsion[len(structure.phi) + i])

    return -loss


def optimize(domain,
             kappa_scalar=8,
             iterations=100, 
             lr=1e-3, 
             lr_decay=1,
             min_lr=1e-10,
             decay_frequency=10,
             normalize_gradients=True,
             momentum=0,
             nesterov=False,
             verbose=1, 
             img_dir=None):
    """
    Optimize structure (phi and psi angles) given a label
    """
    
    # Load predicted data
    
    with open(f'outputs/{domain}.out', 'rb') as f:
        d = pickle.load(f)
    
    distogram, phi, psi = d['distogram'], d['phi'], d['psi']
    
    # remove first phi angle and last psi angle
    # necessary because the first atom we place is Nitrogen and last is Carbon-C
    phi = phi[:, :, :, 1:]
    psi = psi[:, :, :, :-1]
    
    # sample angles from von Mises distribution fitted to each histogram in angleograms
    phi_sample, psi_sample = sample_torsion(phi, psi, kappa_scalar)
    
    # fit continuous von Mises distribution to each histogram in angleograms
    vmphi = fit_vm(phi)
    vmpsi = fit_vm(psi)
    
    # load sequence
    # with open(f'../../data/our_input/sequences/{domain}.fasta') as f:
    #     f.readline()  # fasta header
    #     seq = f.readline()
    
    #seq = domains[f'{domain}'][2]
    with open('1a02F00.fasta') as f:
        f.readline()
        seq = f.readline()
    
    # Create Structure - Model of protein geometry
    structure = Structure(phi_sample, psi_sample, seq)
    s0 = copy(structure)
    # OPTIMIZE, OPTIMIZE, OPTIMIZE
    
    history = []
    min_loss = np.inf
    
    if momentum > 1 or momentum < 0:
        print('Momentum parameter has to be between 0 and 1')
        return
    
    # initialize V for momentum
    V = torch.zeros((len(structure.torsion)))
    
    for i in range(iterations):
        if structure.torsion.grad is not None:
            structure.torsion.grad.zero_()
        
        if nesterov:
            structure.torsion = (structure.torsion + momentum * V).detach().requires_grad_()
            
        L = NLLLoss(structure, distogram, vmphi, vmpsi)
        L.backward()

        if normalize_gradients:
            # normalize gradients
            structure.torsion.grad = (structure.torsion.grad - torch.mean(structure.torsion.grad)) / torch.std(structure.torsion.grad)
        
        # Implementing momentum
        V = momentum * V - lr * structure.torsion.grad
        
        structure.torsion = (structure.torsion + V).detach().requires_grad_()
        
        if verbose is not None:
            if i % verbose == 0:
                print(f'Iteration {i}, Loss: {L.item()}')
                
        history.append([i, L.item()])
        
        if i % decay_frequency == 0 and i > 0:
            lr *= lr_decay
        
        if L.item() < min_loss:
            best_structure = copy(structure)
            min_loss = L.item()
            
        if img_dir is not None:
            structure.visualize_structure('{}/iter_{:04d}.png'.format(img_dir, i))
        
        if lr < min_lr:
            break
            
    return best_structure, s0, min_loss, np.array(history)