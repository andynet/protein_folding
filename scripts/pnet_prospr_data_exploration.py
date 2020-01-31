#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:14:32 2020

@author: tomasla
"""

#%%
import pickle
import numpy as np

#%%
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
# Domain Names from the ProSPr dataset
prospr_train = open_pickle('../data/ProSPr/TRAIN-p_names.pkl')
prospr_test = open_pickle('../data/ProSPr/TEST-p_names.pkl')

prospr_train = prospr_train.ravel()
prospr_test = prospr_test.ravel()

prospr_domains_raw = np.concatenate((prospr_train, prospr_test))
#%%
def get_pnet_names(filepath):
    '''Opens a ProteinNet file and extracts the domain names'''
    names = []
    with open(filepath) as f:
        for line in f:
            if line == '[ID]\n':
                names.append(f.readline().strip())
    return np.array(names)

#%%
### Are there any matches? ###

# The prospr domain names were extracted from CATH. The
#   domain name has the following meaning:

# CHARACTERS 1-4: PDB Code

# CHARACTER 5: Chain Character
#   This determines which PDB chain is represented.
#   Chain characters of zero ('0') indicate that the PDB file has no chain field.

# CHARACTER 6-7: Domain Number
#   The domain number is a 2-figure, zero-padded number (e.g. '01', '02' ... ). 
#   Where the domain number is a double ZERO ('00') this indicates that the 
#   domain is a whole PDB chain with no domain chopping. 


# The ProteinNet Records
# <class>#<pdb_id>_<chain_number>_<chain_id>
# where the quantities inside <> are strings and space-delimited arrays of the 
# form previously described. The <class> field of the ID entry is only present 
# in the validation and test sets, and corresponds to the sequence identity class 
# and CASP class, respectively. For test set entries, the remainder of the ID 
# field only contains the CASP identifier.

#%%
# Reformat prospr names to proteinnet convention
prospr_domains = []

for i, prospr_domain in enumerate(prospr_domains_raw):
    # get domain number. change `0X` numbers to `X` 
    if prospr_domain[-2] == '0':
        domain_number = prospr_domain[-1]
    else:
        domain_number = prospr_domain[-2:]
    
    # reformat prospr domain name into proteinnet convention   
    prospr_domains.append(prospr_domain[:4].upper() + '_' + domain_number + '_' + prospr_domain[4])

prospr_domains = np.array(prospr_domains)

del i, prospr_domain, domain_number
#%%
# Domain Names from ProteinNet12
#pnet12_training30_raw = get_pnet_names('../Data/casp12/training_30')

#%%
# there are many sequences with weird names like: 3vc5_d3vc5a1.
# There is no way of knowing whether the domain name is A or a, which can be two
# different domains. For now I am just going to remove those entries.
# To see the entire raw dataset uncomment the chunk above

# Also, the val/test sets have additional characters in the beggining that need
# to be removed

def proteinnet_names(filepath, valtest = False):
    '''returns numpy array of domain names from proteinnet dataset. Removes 
    wrongly formatted names and for validation and test dataset removes the 
    part of the name before #'''
    
    pnet = get_pnet_names(filepath)
    
    # cut the class if the input is test/validation set
    if valtest:
        for i in range(len(pnet)):    
            pnet[i] = pnet[i].split('#')[1]
    
    # remove entries with weird codes like 3vc5_d3vc5a1
    wrongly_formatted = []
    for i in range(len(pnet)):
        if pnet[i].count('_') != 2:
            wrongly_formatted.append(i)
    
    
    return np.delete(pnet, wrongly_formatted)

#%%
#pnet12_training30 = proteinnet_names('../data/casp12/training_30')

#%%
# Go through all names in ProSPr train and check whether the PDB code is present in pnet
# if record is present increase shared counter by one
# if record is present more than once increse the shared_more_times by one
shared = 0
shared_more_times = 0

for prospr_domain in prospr_domains:
    inside = False
    for pnet_domain in pnet12_training30:
        if pnet_domain.upper()[:4] == prospr_domain.upper()[:4]:
            if inside == False:
                shared += 1
                inside = True
            else:
                shared_more_times += 1
                
## Returns: shared            = 16000
#           shared_more_times = 5729

#%%
# counting how many entries do not have chain field in prospr dataset
no_chain = 0
for i in prospr_domains:
    if i[4] == '0':
        no_chain += 1
        
# returns: 3
#%%

# Unique domain numbers in Prospr:
#['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
#       '11', '12', '13', '14', '15', '17', '18', '19', '20']
        
# Unique domain numbers in ProteinNet
#['0', '1', ..., '87']
#%%
# How many domains match exactly, ie. their PDB code, domain number and chain id

def count_shared_domains(prospr_list, pnet_list = None, pnet_path = None, valtest = False):        
    shared = 0
    shared_domains = []
    if pnet_list is None:
        pnet_list = proteinnet_names(pnet_path, valtest)
    
    for prospr_domain in prospr_list:
        for pnet_domain in pnet_list: 
            if pnet_domain == prospr_domain:
                shared_domains.append(pnet_domain)
                shared += 1
                break
    
    return shared, shared_domains

#%%
    # ProSPr vs ProteinNet12 training30
#p30 = count_shared_domains(prospr_domains[:1000], pnet12_training30[:1000])
# returns: 3227

#%%
# PROSPR vs PROTEINNET 12
train_pnet_paths = ['../data/casp12/training_30', '../data/casp12/training_50', 
              '../data/casp12/training_70', '../data/casp12/training_90', 
              '../data/casp12/training_95', '../data/casp12/training_100']

valtest_pnet_paths = ['../data/casp12/validation', '../data/casp12/testing']

shared_domains = dict()

# train sets
for pnet_path in train_pnet_paths:    
    shared_domains[pnet_path.split('/')[-1]] = count_shared_domains(prospr_domains, pnet_path = pnet_path)

# validation and test set
for pnet_path in valtest_pnet_paths:    
    shared_domains[pnet_path.split('/')[-1]] = count_shared_domains(prospr_domains, pnet_path = pnet_path, valtest = True)

with open('../steps/shared_domains', 'w') as f:
    f.write(str(shared_domains))
    
#%%
# PROSPR vs PROTEINNET 11
train_pnet_paths = ['../data/casp11/training_30', '../data/casp11/training_50', 
              '../data/casp11/training_70', '../data/casp11/training_90', 
              '../data/casp11/training_95', '../data/casp11/training_100']

valtest_pnet_paths = ['../data/casp11/validation', '../data/casp11/testing']

shared_domains = dict()

# train sets
for pnet_path in train_pnet_paths:    
    shared_domains[pnet_path.split('/')[-1]] = count_shared_domains(prospr_domains, pnet_path = pnet_path)

    
# validation and test set
for pnet_path in valtest_pnet_paths:    
    shared_domains[pnet_path.split('/')[-1]] = count_shared_domains(prospr_domains, pnet_path = pnet_path, valtest = True)

with open('../steps/shared_domains11', 'w') as f:
    f.write(str(shared_domains))