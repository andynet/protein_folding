#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:57:51 2020

@author: tomasla

In this file I am going to explore the shared names between proteinnets12 
datasets and prospr. 
"""
#%%
import numpy as np
import ast

#%% PROSPR VS PROTEIN NET 12
with open('../steps/shared_domains') as f:
    shared = ast.literal_eval( f.read())
    
del f

#%%
domains = np.array([])

for i in range(len(shared)):
    domains = np.concatenate([domains, shared[list(shared.keys())[i]][1]])
    
unique_domains12 = np.unique(domains)
#  unique domains


with open('../steps/propspr_pnet12_unique_domains', 'w') as f:
    f.write(str(unique_domains))
    

#%% PROSPR vs PROTEIN NET 11
with open('../steps/shared_domains11') as f:
    shared = ast.literal_eval( f.read())
    
del f

domains = np.array([])
#shared[list(shared.keys())[0]][i]
for i in range(len(shared)):
    domains = np.concatenate([domains, shared[list(shared.keys())[i]][1]])
    
unique_domains11 = np.unique(domains)
#  unique domains

with open('../steps/propspr_pnet12_unique_domains', 'w') as f:
    f.write(str(unique_domains))
    
