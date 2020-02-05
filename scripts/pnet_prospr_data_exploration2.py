#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:57:51 2020

@author: tomasla

In this file I am going to explore the shared names between proteinnets12 
datasets and prospr. 
"""
#%%
import os
os.chdir('/home/tomasla/deeply_thinking_potato/scripts')

import numpy as np
import pandas as pd
import ast

#%% PROSPR VS PROTEIN NET 12
with open('../steps/shared_domains12') as f:
    shared = ast.literal_eval( f.read())
    
del f

domains = np.array([])

for i in range(len(shared)):
    domains = np.concatenate([domains, shared[list(shared.keys())[i]][1]])
    
unique_domains12 = np.unique(domains)
#  unique domains
#%%
#pd.DataFrame(unique_domains12).to_csv('../steps/prospr_proteinnet12_unique_domains')

#%%
# get names
#dfnames = np.array([])
#for i in range(len(shared)):
#    dfname = np.repeat(list(shared.keys())[i], shared[list(shared.keys())[i]][0])
#    dfnames = np.concatenate([dfnames, dfname])
    
#%%
#prospr_pnet12_df = pd.DataFrame({'DF_name':dfnames, 'Domain':domains})
#%%
#prospr_pnet12_df.to_csv('../steps/prospr_vs_proteinnet12_shared_domains.csv')

#%% PROSPR vs PROTEIN NET 11
with open('../steps/shared_domains11') as f:
    shared11 = ast.literal_eval(f.read())
    
del f

domains11 = np.array([])
#shared[list(shared.keys())[0]][i]
for i in range(len(shared11)):
    domains11 = np.concatenate([domains11, shared11[list(shared11.keys())[i]][1]])
    
unique_domains11 = np.unique(domains11)
#  unique domains
#%%
print(np.unique(np.concatenate([unique_domains11, unique_domains12])).shape)

