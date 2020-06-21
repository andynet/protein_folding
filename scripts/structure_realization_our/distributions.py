#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributions - cubic spline fitting + von Mises for torsion angles
"""

import torch
import numpy as np
import pyro
from scipy.stats import vonmises

# cubic spline interpolation
# https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch


def torch_searchsort(x, xs):
    """
    Returns index of a bin where xs belongs, where x is a list of bins
    """
    x = torch.cat((torch.tensor([-10.0]), x, torch.tensor([10000.0])))
    ind = 0
    for i in range(1, len(x) - 1):
        if x[i-1] < xs and xs <= x[i]:
            return i - 1
        

def h_poly_helper(tt):
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=tt[-1].dtype)
    return [sum(A[i, j] * tt[j] for j in range(4)) for i in range(4)]


def h_poly(t):
    tt = [None for _ in range(4)]
    tt[0] = 1
    for i in range(1, 4):
        tt[i] = tt[i - 1] * t
    return h_poly_helper(tt)


def interp(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    I = torch_searchsort(x[1:], xs)
    dx = (x[I + 1] - x[I])
    hh = h_poly((xs - x[I]) / dx)
    return hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx


# Von Mises distribution

# Random number Generator
def randvonmises(anglegram, i, kappa_scalar=8, random_state=1):
    """
    Sample random value from a von Mises distribution fitted to a histogram
    """
    np.random.seed(random_state)
    
    xtorsion = torch.linspace(-np.pi, np.pi, 36)
    
    vmexp = torch.sum(xtorsion * anglegram[0, 1:, 0, i])
    vmvar = torch.sum(anglegram[0, 1:, 0, i] * (xtorsion - vmexp) ** 2)
    
    vmkappa = 1 / vmvar
    
    randvar = vonmises.rvs(kappa=kappa_scalar * vmkappa, loc=vmexp)
    if randvar < -np.pi:
        randvar = 2 * np.pi + randvar
    elif randvar > np.pi:
        randvar = - 2 * np.pi + randvar
    return randvar


def sample_torsion(phi, psi, kappa_scalar=8, random_state=1):
    """
    Samples one value from each von mises distribution fitted to
    each histogram. Kappa is calculated as 1/var * kappa_scalar, to
    make it more narrow
    """
    
    phi_sample = torch.tensor(np.round([randvonmises(phi, i, kappa_scalar, random_state) for i in range(phi.shape[3])], 4))
    psi_sample = torch.tensor(np.round([randvonmises(psi, i, kappa_scalar, random_state) for i in range(psi.shape[3])], 4))
    return phi_sample, psi_sample


# Differentiable von mises
def fit_vm(anglegram, kappa_scalar=8):
    """
    Outputs list of fitted von mises distributions to a specific angleogram
    Each item has a method ".log_prob(x)"
    """
    distros = []
    xtorsion = torch.linspace(-np.pi, np.pi, 36)
    
    for i in range(anglegram.shape[3]):
        vmexp = torch.sum(xtorsion * anglegram[0, 1:, 0, i])
        vmvar = torch.sum(anglegram[0, 1:, 0, i] * (xtorsion - vmexp) ** 2)
        vmkappa = kappa_scalar / vmvar
        
        vm = pyro.distributions.von_mises.VonMises(vmexp, vmkappa)
        distros.append(vm)
    return distros


def calc_moments(distribution):
    """
    Calculate mean and standard deviation of a distribution
    """
    x = torch.linspace(2, 22, 31)
    d_mean = torch.sum(x * distribution)
    d_var = torch.sum(distribution * (x - d_mean) ** 2) 
    
    return d_mean, torch.sqrt(d_var)


def normal_distr(x, mu, sigma, s=1):
    """
    Find probability of a value "x" in a normal distribution with mean
    "mu" and standard deviation "sigma"
    """
    
    return s * 1/(sigma * torch.sqrt(torch.tensor(2 * np.pi))) * torch.exp((-1/2) * ((x - mu) / sigma) ** 2)

def fit_normal(distogram):
    """
    This just calculates means and standard deviation of all histograms
    and saves it into a 3D tensor with depth 2
    """
    L = distogram.shape[1]
    params = torch.empty((3, L, L))
    
    for i in range(L):
        for j in range(L):
            m, s = calc_moments(distogram[:, i, j])
            scalar = torch.max(distogram[:, i, j]) / normal_distr(m, m, s)
            params[0, i, j], params[1, i, j], params[2, i, j] = m, s, scalar
    
    return params