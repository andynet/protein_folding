#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from crops import make_batches
import numpy as np

# open domain lengths
import pickle
with open('../../steps/domain_lengths.pkl', 'rb') as f:
    domain_lengths = pickle.load(f)
    

def make_crops(domain):
    """
    Create 8 different cropping schemes for given domain
    
    Input: 
        domain: string
    Output:
        input_crops : tensor of shape (8 * k, 569, 64, 64)
        output_crops: tensor of shape (8 * k, 64, 64)
        
    where "k" is the number of generated crops for a given domain length 
    """
    L = domain_lengths[domain]
    k = (L // 64 + 1) ** 2
    
    if L % 64 == 0:
        num_crops = 6 * k
        #k0 = k = (L // 64) ** 2  # if the random state is odd
    else:
        num_crops = 8 * k
    
    X = torch.load(f'../../data/our_input/tensors/{domain}_X.pt')
    Y = torch.load(f'../../data/our_input/distance_maps/distance_maps32/{domain}.pt')
    
    input_crops = torch.empty((num_crops, 569, 64, 64))
    output_crops = torch.empty((num_crops, 64, 64))
    
    if L % 64 == 0:
        s = 0
        for i in range(8):
            k_temp = (L // 64 + (1 - i % 2)) ** 2  # k if even, k0 if odd 
            input_crops[s:(s + k_temp)], output_crops[s:(s + k_temp)] = make_batches(X, Y, random_state=1618 + i)
            s += k_temp
    else:
        for i in range(8):
            input_crops[(k * i):(k * (i + 1))], output_crops[(k * i):(k * (i + 1))] = make_batches(X, Y, random_state=1618 + i)
        
    return input_crops, output_crops


def predict_crops(model, input_crops, bins=32, device='cpu'):
    """
    Create predictions for the input crops
    
    Input:
        model       : neural network with method "predict"
        input_crops : tensor of shape (8 * k, 569, 64, 64)
        bins        : int (32 is default), number of bins in resulting distograms
        device      : ("cpu" - default or "cuda")
        
    Output:
        preds: tensor of shape (8 * k, bins, 64, 64)
    """
    
    preds = torch.empty((input_crops.shape[0], bins, 64, 64))
    
    for i in range(input_crops.shape[0] // 8):
        batch_in = input_crops[(8 * i):(8 * (i + 1))]
        
        if device == 'cuda':
            batch_in = batch_in.to('cuda')
            preds[(8 * i):(8 * (i + 1))] = model.predict(batch_in)
            
            del batch_in, batch_out
            torch.cuda.empty_cache()
        else:
            preds[(8 * i):(8 * (i + 1))] = model.predict(batch_in)
    return preds


def unpad_crop(crop, label, bins=32):
    """
    Removes values that were used for padding
    
    Input:
        crop : torch tensor of shape (bins, 64, 64)
        label: torch tensor of shape (bins, 64, 64)
        bins : int (32 is default), number of bins in resulting distograms
    
    Output:
        unpadded crop of shape (bins, width, height) 
    """
    
    non_zero_ind = np.nonzero(label)
    
    if len(non_zero_ind) == 0:
        return crop
    else:
        i0, imax = torch.min(non_zero_ind[:, 0]), torch.max(non_zero_ind[:, 0])
        j0, jmax = torch.min(non_zero_ind[:, 1]), torch.max(non_zero_ind[:, 1])
        
        unpadded = torch.empty((bins, imax - i0 + 1, jmax - j0 + 1))
        
        for (i, j) in non_zero_ind:
            unpadded[:, i - i0, j - j0] = crop[:, i, j]
    return unpadded


def unpad_and_glue(preds, out_crops, L):
    """
    Wrapper function for one unpadding and gluing crops for one domain
    and only one iteration
    
    Input:
        preds    : torch tensor of shape (k, bins, 64, 64)
        out_crops: torch tensor of shape (k, bins, 64, 64)
        L        : domain length
    
    output:
        distogram: torch tensor of shape (bins, L, L)
    """
    
    distogram = torch.empty((32, L, L), dtype=torch.float32)
    k = int(np.sqrt(len(preds)))
    
    i0 = 0
    for i in range(k):
        j0 = 0
        for j in range(k):
            unpadded = unpad_crop(preds[i * k + j], out_crops[i * k + j])
            width, height = unpadded.shape[1:]
            distogram[:, i0:(i0 + width), j0:(j0 + height)] = torch.exp(unpadded)  # log(softmax) -> softmax
            j0 += height
        i0 += width
            
    return distogram


def predict_distogram(model, domain, bins=32, device="cpu"):
    """
    Wrapper function for predicting a probability distribution of distances (distogram)
    
    Input:
        model : neural network with method "predict"
        domain: string
        bins  : int (32 is default), number of bins in resulting distograms
        device: ("cpu" - default or "cuda")
    
    Output:
        distogram: Averaged distogram from several cropping schemes - torch tensor of shape (bins, L, L)
    """
    
    L = domain_lengths[domain]
    distograms = torch.empty((8, 32, L, L))
    
    i, o = make_crops(domain)
    p = predict_crops(model, i, bins, device)
    
    if L % 64 == 0:
        return 'TODO'
    else:
        k = (L // 64 + 1) ** 2
        for i in range(8):
            distograms[i] = unpad_and_glue(p[(i * k):((i + 1) * k)], o[(i * k):((i + 1) * k)], L)
    
    distogram = torch.mean(distograms, dim=0)
        
    return (distogram / torch.sum(distogram, dim=0)).detach()  # normalize histograms


def calc_mean(distogram, bins=32):
    """
    Mean of a distribution: sum(x * p(x))
    """
    x = [0]
    x.extend([2 + 20/30 * i for i in range(bins - 1)])
    x = torch.tensor(x)
    
    L = distogram.shape[1]
    
    mean_distmap = torch.empty((L, L))
    
    for i in range(L):
        for j in range(L):
            mean_distmap[i, j] = torch.sum(distogram[:, i, j] * x)
    return mean_distmap   


def predict_distmap(model, domain, op='argmax'):
    
    if op == 'argmax' or 'mode':
        distogram = _predict(model, domain)
        return torch.argmax(distogram, dim=0)
    elif op == 'mean' or 'average':
        distogram = _predict(model, domain)
        return calc_mean(distogram)
    else:
        return 'Unknown operation'