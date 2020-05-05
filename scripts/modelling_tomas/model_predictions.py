#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:35:26 2020

@author: tomasla
"""

# %%
import torch
from predict import predict_outputs
import matplotlib.pyplot as plt
from Inception_aux import Inception_aux
import pickle

# %% model
model = Inception_aux().to('cuda')

sd = torch.load('../../steps/inception_results/model.pth')
#sd = torch.load('../../steps/inception_results/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(sd['model'])

# %% small protein: L < 64
p1a02 = predict_outputs(model, '1a02F00', device='cuda')

plt.imshow(torch.argmax(p1a02['distogram'], dim=0))

with open('trials/1a02.out', 'wb') as f:
    pickle.dump(p1a02, f)
    
# %% Special case: L % 64 == 0, 1bg6A01 
p1bg6 = predict_outputs(model, '1bg6A01', device='cuda')

plt.imshow(torch.argmax(p1bg6['distogram'], dim=0), cmap = 'viridis_r')

# %% Normal size: L > 64
p16pk = predict_outputs(model, '16pkA01', device='cuda')

plt.imshow(torch.argmax(p16pk['distogram'], dim=0), cmap='viridis_r')

with open('trials/16pk.out', 'wb') as f:
    pickle.dump(p16pk, f)