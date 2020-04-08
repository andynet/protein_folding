#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:56:06 2020

@author: tomasla
"""

# %%
import requests

path = '/faststorage/project/deeply_thinking_potato/'


# %%
def str_to_int(s):
    nums = '0123456789'
    for i in s:
        if i in nums:
            pass
        else:
            return None
    return int(s)


domains = {}
with open(f'{path}data/our_input/cath-domain-seqs-S35.fa') as f:
    while True:
        info = f.readline()
        if info == '':
            break

            # extract domain id and positions
        domain, pos = info.strip().split('|')[2].split('/')
        pos = pos.split('_')  # in case the domain is not continuous

        if len(pos) == 1:  # we only allow continous domains
            start, end = pos[0].strip('-').split('-')
            start, end = str_to_int(start), str_to_int(end)  # check if there is a character in position
            if start is None or end is None:
                f.readline()
            else:
                domains[domain] = [
                    start, end,
                    f.readline().strip()]
        else:
            f.readline()

del info, start, end, f, domain, pos


# %%
def download_pdb(domain):
    protein_name = domain[:4]
    url = 'https://files.rcsb.org/download/' + protein_name + '.pdb'
    myfile = requests.get(url)
    open(f'{path}data/pdbfiles/' + protein_name + '.pdb', 'wb').write(myfile.content)


# %%
for domain in domains:
    download_pdb(domain)
