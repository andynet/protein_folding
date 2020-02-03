#%%
from Bio.PDB import *
import numpy as np
import pandas as pd

def get_CA_coords_from_pdb(domain):
    '''Returns C alpha coordinates of a domain'''
    
    pdbfile = domain[:4].lower() + '.pdb'
    domain_id = domain.split('_')[2]
    
    parser = PDBParser()
    structure = parser.get_structure('', pdbfile)
    
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