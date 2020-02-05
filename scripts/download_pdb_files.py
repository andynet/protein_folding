#%%
import os
os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

import requests
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
#%%
domains = pd.read_csv('../steps/prospr_proteinnet12_unique_domains')
domains = np.array(domains.iloc[:, 1])

#%%
protein_names = []
for i in domains:
    protein_names.append(i[:4].lower())
   
protein_names = np.unique(np.array(protein_names))
#%%
def download_pdb(protein_name):
    url = 'https://files.rcsb.org/download/' + protein_name + '.pdb'
    myfile = requests.get(url)
    open('../data/pdbfiles/' + protein_name + '.pdb', 'wb').write(myfile.content)
    
#%%
#for protein in protein_names:
#    download_pdb(protein)


#%%
# We need to download all PDB files. So I first need to create a list of names that is not going
# to contain the names of files that we already downloaded - `protein_names`

# Domain Names from the ProSPr dataset
prospr_train = open_pickle('../data/ProSPr/TRAIN-p_names.pkl')
prospr_test = open_pickle('../data/ProSPr/TEST-p_names.pkl')

prospr_train = prospr_train.ravel()
prospr_test = prospr_test.ravel()

prospr_domains_raw = np.concatenate((prospr_train, prospr_test))

del prospr_train, prospr_test
#%% Get names of prospr proteins

prospr_proteins = []
for domain in prospr_domains_raw:
    prospr_proteins.append(domain[:4].lower())

prospr_proteins = np.unique(np.array(prospr_proteins))

#%% find the ones we have not downloaded yet

proteins_to_download = np.setdiff1d(prospr_proteins, protein_names)

#%%
#for protein in proteins_to_download:
#    download_pdb(protein)