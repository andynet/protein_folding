#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 01:05:52 2020

@author: andyb
"""

# %%
from Bio import SeqIO
import numpy as np
import argparse
import torch


# %%
def onehot_encode(s):
    '''Convert a sequence into 21xN matrix. For dataloader'''
    seq_profile = np.zeros((len(s), 21), dtype=np.short)
    aa_order = 'ARNDCQEGHILKMFPSTWYVX'
    seq_order = [aa_order.find(letter) for letter in s]

    for i, letter in enumerate(seq_order):
        seq_profile[i, letter] = 1
    return seq_profile


# %%
def get_input(potts_fn, potts_h, potts_j, pssm, seq):

    L = len(seq)
    seq_1hot = onehot_encode(seq)

    arrays = [
        # params j of the potts model flattened from 4D to 3D
        potts_j.reshape((L, L, 21 * 21)),
        # frobenius norms of the potts model
        potts_fn.reshape((L, L, 1)),
        # position repeated horizontally
        np.repeat(np.arange(0, L)[:, np.newaxis], L, axis=1).reshape((L, L, 1)),
        # position repeated vertically
        np.repeat(np.arange(0, L)[np.newaxis, :], L, axis=0).reshape((L, L, 1)),
        # params h of the potts model repeated horizontally
        np.repeat(potts_h[:, np.newaxis, :], L, axis=1),
        # params h of the potts model repeated vertically
        np.repeat(potts_h[np.newaxis, :, :], L, axis=0),
        # one-hot encoded aminoacids repeated horizontally
        np.repeat(seq_1hot[:, np.newaxis, :], L, axis=1),
        # one-hot encoded aminoacids repeated vertically
        np.repeat(seq_1hot[np.newaxis, :, :], L, axis=0),
        # pssm repeated horizontally
        np.repeat(pssm[:, np.newaxis, :], L, axis=1),
        # pssm repeated vertically
        np.repeat(pssm[np.newaxis, :, :], L, axis=0),
        # protein length - constant
        np.full((L, L, 1), L)
    ]

    return (torch.tensor(np.concatenate(arrays, axis=2))
            .permute(2, 0, 1)   # TODO: this could be simplified with
                                # correct arrays creation in the first place
            .to(dtype=torch.float64))


# %%
def get_seq(seq_file):
    fa_recs = list(SeqIO.parse(seq_file, "fasta"))
    assert len(fa_recs) == 1
    return str(fa_recs[0].seq)


# %%
# args = argparse.Namespace()
# args.fn_file = "/faststorage/project/deeply_thinking_potato/data/our_input/potts/2fgcA03_FN.pt"
# args.h_file = "/faststorage/project/deeply_thinking_potato/data/our_input/potts/2fgcA03_h.pt"
# args.j_file = "/faststorage/project/deeply_thinking_potato/data/our_input/potts/2fgcA03_J.pt"
# args.pssm_file = "/faststorage/project/deeply_thinking_potato/data/our_input/pssm/2fgcA03_pssm.pt"
# args.seq_file = "/faststorage/project/deeply_thinking_potato/data/our_input/sequences/2fgcA03.fasta"
# args.output_file = "/faststorage/project/deeply_thinking_potato/data/our_input/tensors/2fgcA03_X.pt"

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_file', required=True)
    parser.add_argument('--h_file', required=True)
    parser.add_argument('--j_file', required=True)
    parser.add_argument('--pssm_file', required=True)
    parser.add_argument('--seq_file', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    fn_tensor = torch.load(args.fn_file)
    h_tensor = torch.load(args.h_file)
    j_tensor = torch.load(args.j_file)
    pssm_tensor = torch.load(args.pssm_file)
    seq = get_seq(args.seq_file)

    X = get_input(potts_fn=fn_tensor.numpy(),
                  potts_h=h_tensor.numpy(),
                  potts_j=j_tensor.numpy(),
                  pssm=pssm_tensor.numpy(),
                  seq=seq)

    torch.save(X, args.output_file)
