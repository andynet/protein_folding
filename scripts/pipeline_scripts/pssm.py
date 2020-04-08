#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line friendly version of pssm_code
"""

# %% Imports
import sys
from pssm_code import make_pssm
import torch

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
