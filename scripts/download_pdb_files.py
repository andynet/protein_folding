#%%
import requests
import numpy as np
import pandas as pd
#%%
domains = pd.read_csv('../steps/prospr_proteinnet12_unique_domains')
domains = np.array(domains.iloc[:, 1])

#%%
protein_names = []
for i in domains:
    protein_names.append(i[:4].lower())
    
#%%
def download_pdb(protein_name):
    url = 'https://files.rcsb.org/download/' + protein_name + '.pdb'
    myfile = requests.get(url)
    open('../data/pdbfiles/' + protein_name + '.pdb', 'wb').write(myfile.content)
    
#%%
for protein in protein_names:
    download_pdb(protein)
    