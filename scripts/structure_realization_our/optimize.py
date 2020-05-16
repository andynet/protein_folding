#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Optimization
"""
# %%
import torch
import numpy as np
from distributions import *
from structure import Structure
from copy import copy
import pickle
import argparse
import os


# %%
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
    for i in range(len(structure.torsion) // 2):
        # torsion angle loss phi
        loss += vmphi[i].log_prob(structure.torsion[i])
        # torsion angle loss psi
        loss += vmpsi[i].log_prob(structure.torsion[len(structure.torsion) // 2 + i])

    return -loss


def optimize(domain,
             structure_path='',
             random_state=1,
             output_dir=None,
             kappa_scalar=8,
             iterations=100, 
             lr=1e-3, 
             lr_decay=1,
             min_lr=1e-10,
             decay_frequency=10,
             normalize_gradients=True,
             momentum=0,
             nesterov=False,
             verbose=-1, 
             img_dir=None,
             initial_structure=False):
    """
    Gradient Descent Algorithm for optimizing Protein Structures
    
    Input:
        domain             : str, domain name\n
        structure_path     : path where Structure is saved. If not provided a new object is created\n
        random_state       : int, for sampling angles from von Mises distribution\n
        output_dir         : path to directory where the output dict should be saved\n
        kappa_scalar       : float, how much should be the von Mises narrowed, default=8\n
        iterations         : int, iterations of the gradient descent algorithm\n
        lr                 : float, learning rate\n
        lr_decay           : float, learning rate decay\n
        decay_frequency    : how often should the lr be adjusted by multiplying by lr_decay\n
        normalize_gradients: bool, default=True\n
        momentum           : float, momentum parameter\n
        nesterov           : bool, default=False, Nesterov Momentum\n
        verbose            : how often should the program print info about losses. Default=iterations/20\n
        img_dir            : dir where intermediate structure plots should be saved, by default this is disabled\n
        initial_structure  : bool, whether initial structure should be saved and outputed
        
    Output:
        best_structure: structure with minimal loss\n
        (s0            : initial structure\n)
        min_loss      : loss of the best structure\n
        history       : history of learning
    """
    
    initial_lr = copy(lr)
    if verbose == -1:
        verbose = int(np.ceil(iterations / 20))
    
    if structure_path == '':
        structure = Structure(domain, random_state, kappa_scalar)
    else:
        with open(f'{structure_path}', 'rb') as f:
            opt = pickle.load(f)
            structure = opt['best_structure']
    
    if initial_structure:
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
        
        if nesterov is True or nesterov == 'True' or nesterov == 'T':
            structure.torsion = (structure.torsion + momentum * V).detach().requires_grad_()
            
        L = NLLLoss(structure, structure.distogram, structure.vmphi, structure.vmpsi)
        
        if L.item() < min_loss:
            best_structure = copy(structure)
            min_loss = L.item()
        
        L.backward()

        if normalize_gradients is True or normalize_gradients == 'True' or normalize_gradients == 'T':
            # normalize gradients
            structure.torsion.grad = (structure.torsion.grad - torch.mean(structure.torsion.grad)) / torch.std(structure.torsion.grad)
        
        # Implementing momentum
        V = momentum * V - lr * structure.torsion.grad
        
        structure.torsion = (structure.torsion + V).detach().requires_grad_()
        
        if verbose != 0:
            if i % verbose == 0 or i == iterations - 1:
                print('Iteration {:03d}, Loss: {:.3f}'.format(i, L.item()))
            
        if L.item() == np.inf or L.item() == -np.inf or L.item() is None:
            print('Loss = inf')
            return
        
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
    
    if output_dir is not None:
        d = {'beststructure':best_structure, 'loss':min_loss, 'history':history}
        
        with open('{}/{}_{:d}_{:.3f}_{:.1f}_{:.1f}_pred.pkl'.format(output_dir, domain, random_state, initial_lr, lr_decay, momentum), 'wb') as f:
            pickle.dump(d, f)
            
    elif initial_structure:
        return best_structure, s0, min_loss, np.array(history)
    else:
        return best_structure, min_loss, np.array(history)


        
