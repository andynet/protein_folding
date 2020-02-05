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
    
    for atom in chain.get_atoms():
        atom_names.append(atom.get_id())
        coords.append(atom.get_coord())
    
    coorddf = pd.DataFrame(coords, index = atom_names, columns = ['X', 'Y', 'Z'])
    
    # filter C-alpha atoms
    coorddf = coorddf[coorddf.index == 'CA']
    
    # for some reason domain 1C8IA01 has two more C-alpha atoms in the end
    if domain == '1c8iA01':
        coorddf = coorddf.iloc[:-2, :]
    
    # 'protein_letters_3to1' is a part of biopython and is loaded automatically
    residues = np.array([i.get_resname() for i in chain.get_residues()])
    
    for i in residues[residues != 'HOH']:
        # if there is unknown residue just append the same as previous
        #if residues[i] in protein_letters_3to1.keys():
        #    primary.append(protein_letters_3to1[residues[i]])
        #else:
        #    primary.append(primary[-1])
        #if i == 'NAG':
        
        # Fixing mistakes by looking at real fasta files
        if domain == '2r01A01' and i == ' CA':
            primary.append('R') # should be arginine
        elif i not in protein_letters_3to1.keys():
            pass
        else:
            primary.append(protein_letters_3to1[i])
        #primary.append(protein_letters_3to1[i])
        
    #for i in residues:
    #    if i == 'UNL':break
    #    else:
    #        primary.append(protein_letters_3to1[i])
    
    #for chain in structure.get_chains():
    #    if chain.id == domain_id:
    #        for atom in chain.get_atoms():
    #            atom_names.append(atom.get_id())
    #            coords.append(atom.get_coord())
    #        break
    #    else:
    #        domain_index += 1
        
    if print_primary:
        print(primary, len(primary))
    #print(np.array(residues))
    #print(np.array(primary))
    
    if len(coorddf) != len(primary):
        return domain, 'Primary != Coords', 'Primary', len(primary), ', Coords: ', len(coorddf)
    coorddf['Primary'] = primary
    
    # get primary sequence correspondent to the coordinates. !!! This is not real
    # sequence !!! There can be few aminoacids missing !!! Full sequences will
    # be downloaded in fasta format in a separate directory
    
    #if primary:
        
    #    primary = np.array(PPBuilder().build_peptides(structure)[domain_index].get_sequence())
    #    coorddf['Primary'] = primary    
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

#%%
#check that everything works in test and train
for i in prospr_test:
    print(i)
    trial = get_CA_coords_from_pdb(i)

# DONE
#%%
for i, domain in enumerate(prospr_train):
    print(i)
    #if i % 500 == 0:
    #    print(i, 'files processed')
    trial = get_CA_coords_from_pdb(domain)
    

#%%
for domain in prospr_test:
    domain_coords = get_CA_coords_from_pdb(domain)
    filename = '../steps/outputs/test_outputs_raw/'+ domain + '.csv'
    domain_coords.to_csv(filename)
    
#%%
for domain in prospr_test:
    domain_coords = get_CA_coords_from_pdb(domain)
    filename = '../steps/outputs/test_outputs_raw/'+ domain + '.csv'
    domain_coords.to_csv(filename)    
    
#%%
    
def resid(domain):
    pdbfile = '../data/pdbfiles/' + domain[:4].lower() + '.pdb'
    domain_id = domain[4]

    parser = PDBParser()
    structure = parser.get_structure('', pdbfile)
    
    chain = structure[0][domain[4]]
    residues = [i.get_resname() for i in chain.get_residues()]
    
    return np.array(residues)[np.array(residues) != 'HOH']