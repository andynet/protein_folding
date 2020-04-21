#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:08:54 2020

@author: andyb

dist, ss, phi, psi = {domain}_Y.pt
dist: tensor of size LxL containing integers from 0 to 32
ss: tensor of size 1xL containing integers from 0 to 7
phi: tensor of size 1xL containing integers from 0 to 36
psi: tensor of size 1xL containing integers from 0 to 36

ss:
H = 1
B = 2
E = 3
G = 4
I = 5
T = 6
S = 7
L = - = 8
missing data = 0

phi, psi:
[-180, -170) = 1
[-170, -160) = 2
...
[-10, 0) = 18
[0, 10) = 19
...
[160, 170) = 35
[170, 180) = 36
missing data = 0

"""

# %%
import numpy as np
import argparse
import torch

# %%
# args = argparse.Namespace()
# args.dist_file = "/faststorage/project/deeply_thinking_potato/data/"\
#     "our_input/distance_maps/distance_maps32/139lA00.pt"
# args.ss_file = "/faststorage/project/deeply_thinking_potato/data/"\
#     "our_input/secondary/139lA00.sec"
# args.phi_file = "/faststorage/project/deeply_thinking_potato/data/"\
#     "our_input/torsion/phi/139lA00_phi.pt"
# args.psi_file = "/faststorage/project/deeply_thinking_potato/data/"\
#     "our_input/torsion/psi/139lA00_psi.pt"
# args.output_file = "/faststorage/project/deeply_thinking_potato/data/"\
#     "our_input/Y_tensors/139lA00_Y.pt"

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--dist_file', required=True)
parser.add_argument('--ss_file', required=True)
parser.add_argument('--phi_file', required=True)
parser.add_argument('--psi_file', required=True)
parser.add_argument('--output_file', required=True)
args = parser.parse_args()

# %%
dist = torch.load(args.dist_file)

# %%
ss = open(args.ss_file, "r").readlines()
assert len(ss) == 1
translation = {
    'H': 1,
    'B': 2,
    'E': 3,
    'G': 4,
    'I': 5,
    'T': 6,
    'S': 7,
    '-': 8,
}
out = np.zeros((len(ss[0])))
for i, letter in enumerate(ss[0]):
    out[i] = translation[letter]
ss = torch.tensor(out)

# %%
# build with assumption that the angles are from [-180, 180]
# (-np.inf, -180) -> 0
# [-180, -170) -> 1
# [-170, -160) -> 2
# ...
# [160, 170) -> 35
# [170, np.inf) -> 36
bins = np.linspace(-180, 170, 36)

phi = torch.load(args.phi_file).numpy()
phi = torch.tensor(np.digitize(phi, bins))

# %%
psi = torch.load(args.psi_file).numpy()
psi = torch.tensor(np.digitize(psi, bins))

# %%
Y = (dist, ss, phi, psi)
torch.save(Y, args.output_file)
