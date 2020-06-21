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
import matplotlib.pyplot as plt

# %%
C_vdW = 1.7

def steric_repulsion(dmap):
    sr = 0
    d = ((C_vdW ** 2 - dmap ** 2) ** 2) / C_vdW
    for i in range(len(dmap) - 1):
        for j in range(i + 1, len(dmap)):
            if dmap[i, j] < C_vdW:
                sr += d[i, j]
    return sr


def NLLLoss(structure, normal=False, distance_threshold=22, steric=False):
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
    if normal or normal == 'True' or normal == 'T':
        for i in range(len(distance_map) - 1):
            for j in range(i + 1, len(distance_map)):
                mu, sigma, s = structure.normal_params[0, i, j], structure.normal_params[1, i, j], structure.normal_params[2, i, j]
                if mu <= distance_threshold:
                    loss += torch.log(max(torch.tensor(0.0001), 
                                      normal_distr(distance_map[i, j], mu, sigma, s)))
     
    else:  # fit cubic spline to histograms
        for i in range(len(distance_map) - 1):
            for j in range(i + 1, len(distance_map)):
                loss += torch.log(max(torch.tensor(0.001),
                                      interp(x, structure.distogram[:, i, j], min(torch.tensor(22), 
                                                                         distance_map[i, j]))))
    if structure.vmphi is not None:
        # TORSION ANGLE POTENTIAL
        for i in range(len(structure.torsion) // 2):
            # torsion angle loss phi
            loss += structure.vmphi[i].log_prob(structure.torsion[i])
            # torsion angle loss psi
            loss += structure.vmpsi[i].log_prob(structure.torsion[len(structure.torsion) // 2 + i])

    if steric:
        loss -= steric_repulsion(distance_map) 
    
    return -loss


def optimize(structure,
             normal=True, # whether normal distr should be fitted instead of a cubic spline
             iterations=100,
             iteration_start=0,
             distance_threshold=1000,
             angle_potential=False,
             lr=1e-3,
             gradient_scaling='sddiv',  # one of ['sddiv', 'normal', 'absmaxdiv']
             momentum=0,
             nesterov=False,
             steric=False,
             verbose=-1,
             img_dir=None,
             figheight=6):
    """
    Gradient Descent Algorithm for optimizing Protein Structures on restarts
    
    Input:
        structure          : object of class Structure
        normal             : bool, whether (scaled) normal distribution should be fitted to the distograms
        iterations         : int, iterations of the gradient descent algorithm\n
        lr                 : float, learning rate\n
        gradient_scaling   : str, type of gradient scaling. Either 'sddiv' (division by standard deviation), 'normal' for standard normalization or "absmaxdiv" for division by the absolute maximum value
        momentum           : float, momentum parameter\n
        nesterov           : bool, default=False, Nesterov Momentum\n
        verbose            : how often should the program print info about losses. Default=iterations/20\n
        
    Output:
        best_structure: structure with minimal loss\n
        min_loss      : loss of the best structure\n
        history       : history of learning
    """
    
    initial_lr = copy(lr)
    if verbose == -1:
        verbose = int(np.ceil(iterations / 20))
    
    # OPTIMIZE, OPTIMIZE, OPTIMIZE
    history = []
    min_loss = np.inf
    
    if momentum > 1 or momentum < 0:
        print('Momentum parameter has to be between 0 and 1')
        return
    
    # initialize V for momentum
    V = torch.zeros((len(structure.torsion)))
    
    for i in range(iteration_start, iteration_start + iterations):
        if structure.torsion.grad is not None:
            structure.torsion.grad.zero_()
        
        if nesterov is True or nesterov == 'True' or nesterov == 'T':
            structure.torsion = (structure.torsion + momentum * V).detach().requires_grad_()
            
        L = NLLLoss(structure, normal, distance_threshold, steric=steric)
        
        loss_minus_th_loss = L.item() - structure.min_theoretical_loss - structure.min_angle_loss
        
        if loss_minus_th_loss < min_loss:
            best_structure = copy(structure)
            min_loss = loss_minus_th_loss
        
        L.backward()
        
        #print(structure.torsion.grad[:5])
        if gradient_scaling == 'normal':
            # normalize gradients
            structure.torsion.grad = (structure.torsion.grad - torch.mean(structure.torsion.grad)) / torch.std(structure.torsion.grad)
        elif gradient_scaling == 'sddiv':
            structure.torsion.grad = structure.torsion.grad / torch.std(structure.torsion.grad)
        elif gradient_scaling == 'absmaxdiv':
            # gradients inside range from -1 to 1
            structure.torsion.grad = structure.torsion.grad / torch.max(torch.abs(structure.torsion.grad))
        
        # Implementing momentum
        V = momentum * V - lr * structure.torsion.grad
        
        structure.torsion = (structure.torsion + V).detach().requires_grad_()
        
        if verbose != 0:
            if i % verbose == 0 or i == iterations - 1:
                print('Iteration {:03d}, Loss: {:.3f}'.format(i, loss_minus_th_loss))
            
        if L.item() == np.inf or L.item() == -np.inf or L.item() is None:
            print('Loss = inf')
            return
        
        history.append([i, loss_minus_th_loss])
        
        if img_dir is not None:
            structure.visualize_structure(figsize=(int(1.5*figheight), figheight), img_path = '{}/iter_{:04d}.png'.format(img_dir, i))
            with torch.no_grad():
                dmap = structure.G()
            dmap += dmap.t()
            
            fig = plt.figure(figsize=(figheight, figheight))
            plt.imshow(dmap, cmap='viridis_r')
            plt.tight_layout()
            plt.savefig('{}/dmap_iter_{:04d}.png'.format(img_dir, i))
            plt.close(fig)
            
            structure.pdb_coords(output_dir = img_dir, filename='struct_iter_{:04d}.pdb'.format(i))
    
    return best_structure, min_loss, history


def optimize_restarts(
        domain,
        domain_path,
        random_state=1,
        normal=True, # whether normal distr should be fitted instead of a cubic spline
        output_dir=None,
        distance_threshold=1000,
        kappa_scalar=1,
        iterations=200, 
        angle_potential=False,
        restarts=5,
        lr=1e-3, 
        lr_decay=0.1,
        gradient_scaling='sddiv',
        momentum=0,
        nesterov=False,
        steric=False,
        verbose=0,
        isdict=False,
        img_dir=None,
        figheight=6):
    
    history = []
    initial_lr = lr
    structure = Structure(domain=domain, domain_path=domain_path, isdict=isdict, random_state=random_state, 
                          kappa_scalar=kappa_scalar, angle_potential=angle_potential, normal=normal)
    for r in range(restarts):
        s, l, h = optimize(structure=structure, 
                           normal=normal,
                           distance_threshold=distance_threshold,
                           iterations=iterations, 
                           iteration_start=r*iterations,
                           angle_potential=angle_potential,
                           lr=lr,
                           gradient_scaling=gradient_scaling,
                           momentum=momentum,
                           nesterov=nesterov,
                           steric=steric,
                           verbose=verbose,
                           img_dir=img_dir, 
                           figheight=figheight
                          )
        lr = lr_decay * lr
        structure = s
        history.extend(h)
    
    if output_dir is not None:
        d = {'beststructure':structure, 'loss':l, 'history':np.array(history)}
        
        with open('{}/{}_{:d}_{:.3f}_{:.1f}_pred.pkl'.format(output_dir, domain, random_state, initial_lr, momentum), 'wb') as f:
            pickle.dump(d, f)
    else:
        return structure, l, np.array(history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Structure optimization with restarts')
    
    parser.add_argument('-d', '--domain', required=True, help='Domain Name')
    parser.add_argument('-dp', '--domainpath', required=True, help='Path to Domain Predictions')

    parser.add_argument('-n', '--numoptim', type=int, metavar='', required=False, help='Number of cluster jobs. Default = 0', default=0)
    parser.add_argument('-nd', '--normal', metavar='', required=False, help='Fit normal distr to distograms. Default = True. If False, 3rd degree spline is fitted', default='True')
    parser.add_argument('-rs', '--randomstate', type=int, metavar='', required=False, help='Random Seed. Default = 1', default=1)
    parser.add_argument('-o', '--outputdir', metavar='', required=False, help='Output Directory', default='./')
    parser.add_argument('-k', '--kappascalar', metavar='', type=float, required=False, help='scale kappa of von Mises distribution. Default = 1', default=1.0)
    parser.add_argument('-i', '--iterations', type=int, metavar='', required=False, help='Number of iterations. Default = 200', default=200)
    parser.add_argument('-ap', '--anglepotential', metavar='', required=False, help='If angle potential should be used for the calculation of Loss. Default = False', default='False')
    parser.add_argument('-r', '--restarts', type=int, metavar='', required=False, help='Number of restarts of the Optimization process from the best previous state with decreased learning rate. Default = 5', default=5)
    parser.add_argument('-lr', '--learningrate', type=float, metavar='', required=False, help='Learning rate. Default = 0.01', default=0.01)
    parser.add_argument('-ld', '--lrdecay', type=float, metavar='', required=False, help='Learning rate decay parameter after each restart. Default = 0.1', default=0.1)
    parser.add_argument('-gr', '--gradientscaling', metavar='', required=False, help='What type of gradient scaling should be applied to the gradient. Options: sddiv (division by standard deviation), normal (standard normalization), absmaxdiv (division by the absolute maximum value). Default = sddiv', default='sddiv')
    parser.add_argument('-m', '--momentum', type=float, metavar='', required=False, help='Momentum parameter. Default = 0', default=0.0)
    parser.add_argument('-nm', '--nesterov', metavar='', required=False, help='Nesterov Momentum. Default = False', default='False')
    parser.add_argument('-v', '--verbose', type=int, metavar='', required=False, help='How often should the program print info about losses. Default = iterations / 20', default=-1)

    args = parser.parse_args()


    if args.numoptim == 0:
        optimize_restarts(
                 domain=args.domain,
                 domain_path=args.domainpath,
                 random_state=args.randomstate,
                 normal=args.normal, 
                 output_dir=args.outputdir,
                 kappa_scalar=args.kappascalar,
                 iterations=args.iterations,
                 angle_potential=args.anglepotential,
                 restarts=args.restarts,
                 lr=args.learningrate,
                 lr_decay=args.lrdecay,
                 gradient_scaling=args.gradientscaling,
                 momentum=args.momentum,
                 nesterov=args.nesterov,
                 verbose=args.verbose
                 )
    else:
        os.system(f"mkdir -p {args.outputdir}/temp_{args.domain}")
        os.system(f"mkdir -p {args.outputdir}/{args.domain}")
        for i in range(args.numoptim):
            if args.numoptim == 1:
                random_state = args.randomstate
            else:
                random_state = i
            with open(f'{args.outputdir}/temp_{args.domain}/{args.domain}_{i}.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('#SBATCH --mem=4g\n')
                f.write('#SBATCH -t 300\n')
                f.write(f'#SBATCH -o ../../steps/garbage/{args.domain}-%j.out\n')
                f.write(f'python3 optimize_restarts.py -d {args.domain} -dp {args.domainpath} -nd {args.normal} -rs {random_state} -o {args.outputdir}/{args.domain} -k {args.kappascalar} -i {args.iterations} -ap {args.anglepotential} -r {args.restarts} -lr {args.learningrate} -ld {args.lrdecay} -m {args.momentum} -nm {args.nesterov} -v {args.verbose}')
            os.system(f"sbatch {args.outputdir}/temp_{args.domain}/{args.domain}_{i}.sh")
            
