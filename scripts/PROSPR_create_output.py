#%%
import os 
os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

import numpy as np

import pickle
import matplotlib.pyplot as plt

def open_pickle(filepath):
    '''Opens ProSPr pickle file and extracts the content'''
    objects = []
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return np.array(objects)

#%%
bins = open_pickle('../data/ProSPr/name2bins.pkl')
binexample = bins[0][list(bins[0].keys())[0]]

del bins

#%%
# Bins contain integer values - Bin Ids from 1 to 64
# 0 indicates a missing bin
# ProSpr guys had 62 bins in range 2-22 Angstrom, 1 bin for distances larger
# than 22 A and 1 bin for missing values (0 in our case)

