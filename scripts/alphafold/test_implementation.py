#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:31:10 2020

@author: andyb
"""

# %%
from models import AlphaFold
import itertools
import torch
import pandas as pd
# import time   # calculate how long one update takes
import gc

B_to_MB = 1048576


# %%
def to_info(item, device):
    start = torch.cuda.memory_stats(device=device)['active_bytes.all.current']
    item = item.to(device=device)
    end = torch.cuda.memory_stats(device=device)['active_bytes.all.current']
    mb_used = (end - start) / B_to_MB
    return item, mb_used


# %%
def gpu_mem():
    return torch.cuda.memory_stats(device=device)['active_bytes.all.current']


# %%
device = torch.device('cuda')

batch_sizes = [6, 8]  # range(4, 16, 2)
crop_sizes = [64]  # range(32, 64, 4)
RESNET_depths = [210]  # range(64, 256, 16)

# %%
df = pd.DataFrame(columns=['batch', 'crop', 'depth', 'memory'])
print(f'batch\tcrop\tdepth\tmemory')
i = 0

for batch_size, crop_size, RESNET_depth in itertools.product(batch_sizes, crop_sizes, RESNET_depths):

    X = torch.rand([batch_size, 675, crop_size, crop_size])
    Y = torch.randint(64, (batch_size, crop_size, crop_size))

    model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64, RESNET_depth=RESNET_depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model, model_mb = to_info(model, device)
    Y, Y_mb = to_info(Y, device)
    X, X_mb = to_info(X, device)

    with torch.no_grad():
        before = gpu_mem()
        loss1 = model.score(X, Y, criterion)
        after = gpu_mem()
        loss1_mb = (after - before) / B_to_MB

    before = gpu_mem()
    model.fit(X, Y, batch_size, criterion, optimizer)
    after = gpu_mem()
    fit_mb = (after - before) / B_to_MB

    with torch.no_grad():
        before = gpu_mem()
        loss2 = model.score(X, Y, criterion)
        after = gpu_mem()
        loss2_mb = (after - before) / B_to_MB

    # # %%
    # print(f'Model: {model_mb}MB  X: {X_mb}MB  Y: {Y_mb}MB  '
    #       f'loss1_mb: {loss1_mb}MB  fit_mb: {fit_mb}MB  loss2_mb: {loss2_mb}MB')

    # print(torch.cuda.memory_summary(device=device, abbreviated=True))
    memory = torch.cuda.max_memory_reserved(device=device) / B_to_MB
    print(f'{batch_size}\t{crop_size}\t{RESNET_depth}\t{memory}')
    tmp = pd.DataFrame.from_records([[batch_size, crop_size, RESNET_depth, memory]],
                                    columns=['batch', 'crop', 'depth', 'memory'])
    df = df.append(tmp)
    i += 1

    del model, X, Y, loss1, loss2, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    # print(torch.cuda.memory_summary(device=device, abbreviated=True))

# %%
# df.plot.line(subplots=True)
