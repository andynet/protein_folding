#%%
import os
os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

from Bio.PDB import *
import numpy as np
import pandas as pd

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

def get_CA_coords_from_pdb(domain, print_primary = False):
    '''Returns C alpha coordinates of a domain'''
    
    pdbfile = '../data/pdbfiles/' + domain[:4].lower() + '.pdb'
    domain_id = domain[4]
    
    parser = PDBParser()
    structure = parser.get_structure('', pdbfile)
    
    # get coordinates
    domain_index = 0
    
    coords = []
    atom_names = []
    primary = []
    
    chain = structure[0][domain_id]
    
    # get id of first atom
    for atom in chain.get_atoms():
        aid = atom.get_full_id()[3][1] - 1
        break
    
    for atom in chain.get_atoms():
        #if atom.get_full_id()[3][0] != ' ':
        #if atom.get_full_id()[3][1] == aid + 1:
        #    atom_names.append(atom.get_id())
        #    coords.append(atom.get_coord())
        #    aid = atom.get_full_id()[3][1]
        #else:
        #    pass
        #else:
        
    
    coorddf = pd.DataFrame(coords, index = atom_names, columns = ['X', 'Y', 'Z'])
    
    # filter C-alpha atoms
    coorddf = coorddf[coorddf.index == 'CA']
        
    # for some reason domain 1C8IA01 has two more C-alpha atoms in the end
    #if domain == '1c8iA01':
    #    coorddf = coorddf.iloc[:-2, :]
    #if domain == '3lyeA00': # 3 more atoms in the end
    #    coorddf = coorddf.iloc[:-3, :]
    #if domain == '3vgfA03': # 3 more atoms in the end
    #    coorddf = coorddf.iloc[:-1, :]
    
    # get primary sequence correspondent to the coordinates. !!! This is not real
    # sequence !!! There can be few aminoacids missing !!! Full sequences will
    # be downloaded in fasta format in a separate directory
    
    # 'protein_letters_3to1' is a part of biopython and is loaded automatically
    residues = np.array([i.get_resname() for i in chain.get_residues()])
    
    for i in residues[residues != 'HOH']:
        
        # Fixing mistakes by looking at real fasta files
        if domain == '2r01A01' and i == ' CA':
            primary.append('R') # should be arginine
        elif i not in protein_letters_3to1.keys():
            break#pass
        else:
            primary.append(protein_letters_3to1[i])
        #primary.append(protein_letters_3to1[i])
        
        
    if print_primary:
        print(primary, len(primary))
        return coorddf
    
    # few domains have extra coordinates that are marked as Calpha. Thus I 
    # will remove them. See above, I tried doing it manually, but its just not
    # worth it
     
    if len(coorddf) != len(primary):
        #print('Primary != Coords', 'Primary', len(primary), ', Coords: ', len(coorddf))
        return 'Primary != Coords', 'Primary', len(primary), ', Coords: ', len(coorddf)
        #coorddf = coorddf.iloc[:len(primary),:]
    coorddf['Primary'] = primary
    
    return coorddf
#%%
prospr_train = open_pickle('../data/ProSPr/TRAIN-p_names.pkl')
prospr_test = open_pickle('../data/ProSPr/TEST-p_names.pkl')

prospr_train = prospr_train.ravel()
prospr_test = prospr_test.ravel()
#%%
# There are only three domains with no chain, ie their domain id 0. This gives errors
# and I really dont feel like its worth fixing

no_chain_ind = []
for i, domain in enumerate(prospr_train):
    if domain[4] == '0':
        no_chain_ind.append(i)
        
prospr_train = np.setdiff1d(prospr_train, prospr_train[no_chain_ind])

# Biopython had problems openning these two files, thus i removed them
prospr_train = np.delete(prospr_train, np.array([np.where(prospr_train == '2me8A00'),
                                                 np.where(prospr_train == '3cmyA00')]))

#%%
#check that everything works in test and train
for ind, i in enumerate(prospr_test):
    print(ind, i)
    trial = get_CA_coords_from_pdb(i)
    if type(trial) == tuple:
        print(i, trial)
        break

# DONE
#%%
#for i, domain in enumerate(prospr_train):
#    print(i)
    #if i % 500 == 0:
    #    print(i, 'files processed')
#    trial = get_CA_coords_from_pdb(domain)
# DONE    
#%%
for domain in prospr_test:
    domain_coords = get_CA_coords_from_pdb(domain)
    filename = '../steps/outputs/test_outputs_raw/'+ domain + '.csv'
    domain_coords.to_csv(filename)
    #if domain == prospr_test[0]:
        #break
    
#%%
for domain in prospr_train:
    domain_coords = get_CA_coords_from_pdb(domain)
    filename = '../steps/outputs/train_outputs_raw/'+ domain + '.csv'
    domain_coords.to_csv(filename)    
    #if domain == prospr_train[0]:
        #break
    
#%%
    
def resid(domain):
    pdbfile = '../data/pdbfiles/' + domain[:4].lower() + '.pdb'
    domain_id = domain[4]

    parser = PDBParser()
    structure = parser.get_structure('', pdbfile)
    
    chain = structure[0][domain[4]]
    residues = [i.get_resname() for i in chain.get_residues()]
    
    return np.array(residues)[np.array(residues) != 'HOH']