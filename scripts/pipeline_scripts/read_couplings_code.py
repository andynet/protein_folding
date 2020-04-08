import numpy as np


def read_couplings(filename):
    with open(filename) as f:
        raw = f.readlines()

    L = int((1 + np.sqrt(1 + 8 * len(raw))) / 2)  # sequence length
    couplings = np.zeros((L, L))

    counter = 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            couplings[i, j] = float(raw[counter].strip().split()[-1])
            counter += 1

    couplings += couplings.T
    return couplings
