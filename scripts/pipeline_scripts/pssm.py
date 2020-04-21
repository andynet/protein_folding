#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line friendly version of pssm_code

File contains Function for reading psiblast ascii pssm output and transforms
it into matrix/tensor form
"""

# %% Imports
import sys
import torch
import numpy as np


def make_pssm(filepath):
    pssm = []
    with open(filepath) as f:

        f.readline()
        f.readline()
        ln = f.readline()
        # raw.append(ln.strip().split())  # aminoacid order
        # ARNDCQEGHILKMFPSTWYV
        for ln in f:
            if ln == '\n':
                break
            pssm.append([int(i) for i in ln.strip().split()[2:22]])

    return np.array(pssm)


# %%
if len(sys.argv) == 1 or sys.argv[1] == '--help' or sys.argv[1] == '-h':
    sys.exit("""
    # Help
    Transform ascii psiblast pssm file to csv of dimensions (Lx20)

    python pssm.py <psiblast_ascii_pssm> <output_file>
    """)

else:
    pssm = make_pssm(sys.argv[1])
    torch.save(torch.from_numpy(pssm), sys.argv[2])
