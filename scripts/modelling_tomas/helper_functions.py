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
        Y = torch.load(f'../../data/our_input/Y_tensors/{self.domains[idx]}_Y.pt')
        i, o = make_batches(X, Y, random_state=self.random_state)
        return i, o, start, end

    
def load_crops(domains, lengths, random_state, num_workers):
    
    domains_lengths = [lengths[i] for i in domains]
    crops = ncrops(domains_lengths, random_state)
    num_crops = sum(crops)
    
    dataset = LoadDomain(domains, crops, random_state=random_state)
    loader = data.DataLoader(dataset, num_workers=num_workers)
    
    X = torch.empty((num_crops, 569, 64, 64), dtype=torch.float32)
    #Y = torch.empty((num_crops, 64, 64), dtype=torch.long)
    Y = torch.empty((num_crops, 70, 64), dtype=torch.long)
    
    for c in loader:
        i, o, start, end = c
        X[start:end] = i[0]
        Y[start:end] = o[0]
    return X, Y


def evaluate(model, domains, lengths, random_state, num_workers, BATCH_SIZE=8, device='cuda'):
    X, Y = load_crops(domains, lengths, random_state, num_workers)
    
    p_dmat = torch.empty((BATCH_SIZE * (Y.shape[0] // BATCH_SIZE), 32, 64, 64))
    
    p_sec_i = torch.empty((BATCH_SIZE * (Y.shape[0] // BATCH_SIZE), 9, 1, 64))
    p_sec_j = torch.empty((BATCH_SIZE * (Y.shape[0] // BATCH_SIZE), 9, 1, 64))
    
    p_phi_i = torch.empty((BATCH_SIZE * (Y.shape[0] // BATCH_SIZE), 37, 1, 64))
    p_phi_j = torch.empty((BATCH_SIZE * (Y.shape[0] // BATCH_SIZE), 37, 1, 64))
    
    p_psi_i = torch.empty((BATCH_SIZE * (Y.shape[0] // BATCH_SIZE), 37, 1, 64))
    p_psi_j = torch.empty((BATCH_SIZE * (Y.shape[0] // BATCH_SIZE), 37, 1, 64))
    
    if device == 'cuda':
                
        for b in range(X.shape[0] // BATCH_SIZE):
            batch_ind = np.arange(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
            batch_in = X[batch_ind, :, :, :]
                    
            batch_in = batch_in.to("cuda")
                    
            with torch.no_grad():
                batch_preds = model.predict(batch_in)
                
            p_dmat[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds[0]
            
            p_sec_i[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds[1]
            p_sec_j[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds[2]
            
            p_phi_i[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds[3]
            p_phi_j[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds[4]
            
            p_psi_i[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds[5]
            p_psi_j[(b * BATCH_SIZE):(b + 1) * BATCH_SIZE] = batch_preds[6]
                    
            del batch_in, batch_preds
            torch.cuda.empty_cache()
        
        eval_ind = torch.arange((X.shape[0] // BATCH_SIZE) * BATCH_SIZE)
        loss = 10 * F.nll_loss(p_dmat, Y[eval_ind, :64, :]).item()\
                  + F.nll_loss(p_sec_i, Y[eval_ind, 64:65, :]) + F.nll_loss(p_sec_j, Y[eval_ind, 65:66, :])\
                  + F.nll_loss(p_phi_i, Y[eval_ind, 66:67, :]) + F.nll_loss(p_phi_j, Y[eval_ind, 67:68, :])\
                  + F.nll_loss(p_psi_i, Y[eval_ind, 68:69, :]) + F.nll_loss(p_psi_i, Y[eval_ind, 69:, :])
    else:
        print('I give up')
        return
        #preds = model.predict(X)
        #loss = F.nll_loss(preds, Y).item()
        
    return loss


def evaluate_distmap(model, domains, lengths, random_state, num_workers, BATCH_SIZE=8, device='cuda'):
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