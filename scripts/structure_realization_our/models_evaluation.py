#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:17:43 2020

@author: tomasla
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc

font = {'size'   : 15}

rc('font', **font)

# %%
df = pd.read_csv('../steps/models_lddt.csv', index_col = 0)

# %%
df = pd.melt(df, var_name='Model', value_name='DlDDT')

# %%
fig = plt.figure(figsize=(12, 7))
sns.boxplot('Model', 'DlDDT', data=df)
plt.tight_layout()
plt.xlabel('')


plt.savefig('../plots/models_lddt_nice.png')