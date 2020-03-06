#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:46:04 2020

@author: andyb
"""

# %%
from alphafold_v2 import AlphaFold
import matplotlib.pyplot as plt
import torch


# %%
path = '/faststorage/project/deeply_thinking_potato/data/prospr/overfitted_model.pt'
model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
                  RESNET_depth=64).to(dtype=torch.float64)
model.load_state_dict(torch.load(path))

# %%
domain = '2lhfA00'
path = '/faststorage/project/deeply_thinking_potato/data/prospr/tensors3/{}.pt'
X, Y = torch.load(path.format(domain))
crop_size = 64
seq_length = X.shape[2]
crop_times = seq_length // crop_size

# %%
# tmp = plt.colormaps()

# %%
i_offset = 0
j_offset = 0
X, _ = crop_data(X, Y, i_offset, j_offset, crop_size)

# %%
i_offset = seq_length - crop_size * crop_times
j_offset = seq_length - crop_size * crop_times
X, _ = crop_data(X, Y, i_offset, j_offset, crop_size)

# %%
with torch.no_grad():
    predictions = model.predict(X)
    predictions = predictions.argmax(dim=1)
    tmp = torch.zeros(crop_size * crop_times, crop_size * crop_times)

# %%
n = 0
i, j = 0, 0
for i in range(crop_times):
    for j in range(crop_times):
        tmp[i * 64:i * 64 + 64, j * 64:j * 64 + 64] = predictions[n]
        n += 1
# %%
plt.subplot(1, 2, 1)
plt.imshow(tmp, cmap='viridis_r')
plt.subplot(1, 2, 2)
plt.imshow(Y, cmap='viridis_r')
