#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:06:13 2020

@author: andyb

dist:
bins32 = np.concatenate(([0], np.linspace(2, 22, 30), [1000]))
[0.0, 2.0, 2.69, 3.38, 4.07, 4.76, 5.45, 6.14, 6.83, 7.52, 8.21, 8.9, 9.59,
 10.28, 10.97, 11.66, 12.34, 13.03, 13.72, 14.41, 15.1, 15.79, 16.48, 17.17,
 17.86, 18.55, 19.24, 19.93, 20.62, 21.31, 22.0, 1000.0]

0 = (-inf, 0.0)
1 = [0.0, 2.0)
2 = [2.0, 2.69)
...
10 = [7.52, 8.21)
...

rr file is 1-indexed
i < j

ss file:
>1eazA.ss
CCEEEEEEEEECCCCCCCEEEEEEEECCCEEEEEECCCCCCECEEEECCCCEEEEEECCCCHCCCCCEEEEECCCCEEEEECCCHHHHHHHHHHHHHHHHHCC

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

https://books.google.dk/books?id=Ki8BE-R7WsgC&pg=PA140
(G, H, I) -> H
(E, B) -> E
(T, S, C) -> C
"""

# %%
from torch.nn.functional import softmax
import argparse
import torch


# %%
def write(contacts, fa_file, rr_file):

    with open(fa_file) as f:
        fa = f.readlines()

    with open(rr_file, 'w') as f:
        f.write(f'{fa[1]}\n')
        for i in range(contacts.shape[0]):
            for j in range(i, contacts.shape[1]):
                if contacts[i, j]:
                    f.write(f'{i + 1} {j + 1} 0 8 1\n')


def write_ss(ss_seq, fa_file, ss_file):
    with open(fa_file) as f:
        fa = f.readlines()

    with open(ss_file, 'w') as f:
        f.write(f'{fa[0]}')
        f.write(f'{ss_seq}\n')


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--in_file', required=True)
parser.add_argument('--fa_file', required=True)
parser.add_argument('--rr_file', required=True)
parser.add_argument('--ss_file', required=True)
args = parser.parse_args()

# %%
# args = argparse.Namespace()
# # args.in_file = "/faststorage/project/deeply_thinking_potato/data/our_input/Y_tensors/1bkbA02.pred.pt"
# args.in_file = "/faststorage/project/deeply_thinking_potato/data/our_input/Y_tensors/1bkbA02.real.pt"
# args.fa_file = "/faststorage/project/deeply_thinking_potato/data/our_input/sequences/1bkbA02.fasta"

# args.rr_file = "/faststorage/project/deeply_thinking_potato/data/our_input/contacts/1bkbA02.pred.rr"
# args.ss_file = "/faststorage/project/deeply_thinking_potato/data/our_input/secondary_structures/1bkbA02.pred.ss"

# %%
threshold = 0.5
high_bin = 11       # first bin predicting non-contact
case = args.in_file.split('.')[-2]
translation = {1: 'H', 2: 'E', 3: 'E', 4: 'H', 5: 'H', 6: 'C', 7: 'C', 8: 'C'}

# %%
Y = torch.load(args.in_file)

# %%
if case == 'pred':
    contacts = softmax(Y[0], dim=1)
    contacts = (contacts + contacts.permute((0, 1, 3, 2))) / 2
    contacts = contacts[:, 0:high_bin, :, :].sum(dim=1).squeeze().numpy()
    contacts = contacts > threshold

    ss_codes = Y[1].argmax(dim=1).squeeze().numpy() + 1
    ss_letters = [translation[x] for x in list(ss_codes)]
    ss_seq = ''.join(ss_letters)

elif case == 'real':
    contacts = Y[0].numpy()
    contacts = contacts < high_bin

    ss_codes = Y[1].numpy()
    ss_letters = [translation[x] for x in list(ss_codes)]
    ss_seq = ''.join(ss_letters)


write(contacts, args.fa_file, args.rr_file)
write_ss(ss_seq, args.fa_file, args.ss_file)
