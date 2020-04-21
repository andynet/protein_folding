#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training and Evaluation Helper Functions
"""

from torch.utils import data
import torch
import numpy as np
from crops import make_batches
import torch.nn.functional as F

def ncrops(lengths, random_state=1):
    nc = []
    
    for i in lengths:
        if i % 64 == 0:
            if random_state % 2 == 0:
                nc.append((i // 64 + 1) ** 2)
            else:
                nc.append((i // 64) ** 2)
        elif i < 64:
            nc.append(1)
        else:
            nc.append((i // 64 + 1) ** 2)
    return nc


class LoadDomain(data.Dataset):
    def __init__(self, domains, crop_list, random_state=1):
        'Initialization'
        self.domains = domains
        self.random_state = random_state
        self.crop_list = crop_list
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.domains)

    def __getitem__(self, idx):
        'Generates one sample of data'
        start = sum(self.crop_list[:idx])
        end = start + self.crop_list[idx]        
        
        X = torch.load(f'../../data/our_input/tensors/{self.domains[idx]}_X.pt')
        Y = torch.load(f'../../data/our_input/distance_maps/distance_maps32/{self.domains[idx]}.pt')
        i, o = make_batches(X, Y, random_state=self.random_state)
        return i, o, start, end

    
def load_crops(domains, lengths, random_state, num_workers):
    
    domains_lengths = [lengths[i] for i in domains]
    crops = ncrops(domains_lengths, random_state)
    num_crops = sum(crops)
    
    dataset = LoadDomain(domains, crops, random_state=random_state)
    loader = data.DataLoader(dataset, num_workers=num_workers)
    
    X = torch.empty((num_crops, 569, 64, 64), dtype=torch.float32)
    Y = torch.empty((num_crops, 64, 64), dtype=torch.long)
    
    for c in loader:
        i, o, start, end = c
        X[start:end] = i[0]
        Y[start:end] = o[0]
    return X, Y


def evaluate(model, domains, lengths, random_state, num_workers, BATCH_SIZE=8, device='cuda'):
    X, Y = load_crops(domains, lengths, random_state, num_workers)
    
    preds = torch.empty((BATCH_SIZE*(Y.shape[0] // BATCH_SIZE), 32, 64, 64))
    if device == 'cuda':
                
        for b in range(X.shape[0] // BATCH_SIZE):
            batch_ind = np.arange(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
            batch_in = X[batch_ind, :, :, :]
                    
            batch_in = batch_in.to("cuda")
                    
            with torch.no_grad():
                batch_preds = model.predict(batch_in)
                    
            preds[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds.to('cpu')
                    
            del batch_in, batch_preds
            torch.cuda.empty_cache()
                
        loss = F.nll_loss(preds, 
                          Y[torch.arange((X.shape[0] // BATCH_SIZE) * BATCH_SIZE),:,:]).item()
    else:
        preds = model.predict(X)
        loss = F.nll_loss(preds, Y).item()
        
    return loss