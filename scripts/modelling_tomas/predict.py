#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction function for entire domain. The main function is `predict_distmap`
that takes as input:
    model
    domain
    iterations
This function generate `iterations` number of cropping schemes, predict each crop, 
glues them together and finally takes average of all predicted distance maps.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from crops import make_batches


def unpad_crop(crop, label):
    non_zero_ind = np.nonzero(label)
    
    if len(non_zero_ind) == 0:
        return crop
    else:
        i0, imax = torch.min(non_zero_ind[:, 0]), torch.max(non_zero_ind[:, 0])
        j0, jmax = torch.min(non_zero_ind[:, 1]), torch.max(non_zero_ind[:, 1])
        
        unpadded = torch.empty((imax - i0 + 1, jmax - j0 + 1))
        
        for (i, j) in non_zero_ind:
            unpadded[i - i0, j - j0] = crop[i, j]
    return unpadded


def predict_and_glue(model, domain, random_state=1):
    
    X = torch.load(f'../../data/our_input/tensors/{domain}_X.pt')
    Y = torch.load(f'../../data/our_input/distance_maps/distance_maps32/{domain}.pt')
    
    L = Y.shape[1]

    if L % 64 == 0 and random_state % 2 != 0:
        k = L // 64
    else:
        k = L // 64 + 1
    
    i, o = make_batches(X, Y, random_state=random_state)
    preds = model.predict(i)
       
    dist_map = torch.empty((L, L), dtype=torch.long)
    
    i0 = 0
    for i in range(k):
        j0 = 0
        for j in range(k):
            pred = torch.argmax(preds[i * k + j], dim=0)
            unpadded = unpad_crop(pred, o[i * k + j])
            imax, jmax = unpadded.shape
            dist_map[i0:(i0 + imax), j0:(j0 + jmax)] = unpadded
            j0 += jmax
        i0 += imax
    return dist_map


def predict_distmap(model, domain, iterations=4):
    
    distmaps = torch.tensor([], dtype=torch.long)
    for i in range(iterations):
        pred = predict_and_glue(model, domain, random_state=1618+i)
        distmaps = torch.cat((distmaps, pred.view(1, pred.shape[0], pred.shape[0])))
    
    return torch.round(torch.mean(distmaps.to(torch.float), axis=0)).to(torch.long)