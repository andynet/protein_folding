# NOT DONE. NEED PICKLES TO DO FULL CHECKS

#%%
import os 
os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

import numpy as np

import pickle

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
# This file contains functions that check whether all Potts models from ProSPR
# were downloaded and also checks whether we have coordinates for each domain

#%%
prospr_train = open_pickle('../data/ProSPr/TRAIN-p_names.pkl')
prospr_test = open_pickle('../data/ProSPr/TEST-p_names.pkl')

prospr_train = prospr_train.ravel()
prospr_test = prospr_test.ravel()

prospr_domains_raw = np.concatenate((prospr_train, prospr_test))

del prospr_train, prospr_test

#%%
protein_names = []
for i in prospr_domains_raw:
    protein_names.append(i[:4].lower())
   
protein_names = np.unique(np.array(protein_names))

#%%
for file in os.listdir('../data/pdbfiles'):
    if file[:4] not in protein_names:
        print(file)
        break
    
############
## PASSED ##
############
        
#%%
# now for every domains I need to look inside the file to see whether it is
# included
        
from Bio.PDB import *

def check_domain(domain):
    '''checks whether domain and it primary sequence is included'''
    
    pdbfile = '../data/pdbfiles/' + domain[:4].lower() + '.pdb'
    domain_id = domain[4]
    
    parser = PDBParser()
    structure = parser.get_structure('', pdbfile)
    
    # check whether domain is inside
    domain_inside = False
    for chain in structure.get_chains():
        if chain.id == domain_id:
            domain_inside = True
            break
    
    
    coords = []
    atom_names = []
    for chain in structure.get_chains():
        if chain.id == domain_id:
            for atom in chain.get_atoms():
                atom_names.append(atom.get_id())
                coords.append(atom.get_coord())
            break
    coorddf = pd.DataFrame(coords, index = atom_names, columns = ['X', 'Y', 'Z'])
    
    return coorddf[coorddf.index == 'CA']
