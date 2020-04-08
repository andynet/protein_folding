'''
File contains Function for reading psiblast ascii pssm output and transforms
it into matrix/tensor form
'''

import numpy as np


def make_pssm(filepath):
    pssm = []
    with open(filepath) as f:

        f.readline()
        f.readline()
        ln = f.readline()
        # raw.append(ln.strip().split())  # aminoacid order
        # ARNDCQEGHILKMFPSTWYV
        while True:
            ln = f.readline()
            if ln == '\n':
                break
            pssm.append([int(i) for i in ln.strip().split()[2:22]])

    return np.array(pssm)
