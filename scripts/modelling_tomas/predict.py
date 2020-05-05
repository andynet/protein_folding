#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
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
        k0 = (L // 64) ** 2  # if the random state is odd
        num_crops = 4 * k + 4 * k0
        
    else:
        num_crops = 8 * k
    
    X = torch.load(f'../../data/our_input/tensors/{domain}_X.pt')
    Y = torch.load(f'../../data/our_input/Y_tensors/{domain}_Y.pt')
    
    input_crops = torch.empty((num_crops, 569, 64, 64))
    output_crops = torch.empty((num_crops, 70, 64))
    
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
    
    p_dist = torch.empty((input_crops.shape[0], bins, 64, 64))
    p_sec_i, p_sec_j = torch.empty((input_crops.shape[0], 9, 1, 64)), torch.empty((input_crops.shape[0], 9, 1, 64))
    p_phi_i, p_phi_j = torch.empty((input_crops.shape[0], 37, 1, 64)), torch.empty((input_crops.shape[0], 37, 1, 64))
    p_psi_i, p_psi_j = torch.empty((input_crops.shape[0], 37, 1, 64)), torch.empty((input_crops.shape[0], 37, 1, 64))
    
    for i in range(input_crops.shape[0] // 4):
        i0, imax = 4 * i, 4 * (i + 1)
        batch_in = input_crops[i0:imax]
        
        if device == 'cuda':
            batch_in = batch_in.to('cuda')
            p = model.predict(batch_in)
            
            del batch_in
            torch.cuda.empty_cache()
        else:
            p = model.predict(batch_in)
        
        p_dist[i0:imax] = p[0].to('cpu')
        p_sec_i[i0:imax], p_sec_j[i0:imax] = p[1].to('cpu'), p[2].to('cpu')
        p_phi_i[i0:imax], p_phi_j[i0:imax] = p[3].to('cpu'), p[4].to('cpu')
        p_psi_i[i0:imax], p_psi_j[i0:imax] = p[5].to('cpu'), p[6].to('cpu')
    
    return p_dist, (p_sec_i, p_sec_j, p_phi_i, p_phi_j, p_psi_i, p_psi_j)


def unpad_crop(crop, label, bins=32):
    """
    Removes values that were used for padding
    
    Input:
        crop : torch tensor of shape (bins, 64, 64)
        label: torch tensor of shape (64, 64)
        bins : int (32 is default), number of bins in resulting distograms
    
    Output:
        unpadded crop of shape (bins, width, height) 
    """
    
    non_zero_ind = np.nonzero(label[:64, :])
    
    if len(non_zero_ind) == 0:
        return crop
    else:
        i0, imax = torch.min(non_zero_ind[:, 0]), torch.max(non_zero_ind[:, 0])
        j0, jmax = torch.min(non_zero_ind[:, 1]), torch.max(non_zero_ind[:, 1])
        
        unpadded = torch.empty((bins, imax - i0 + 1, jmax - j0 + 1))
        
        for (i, j) in non_zero_ind:
            unpadded[:, i - i0, j - j0] = crop[:, i, j]
    return unpadded


def unpad_aux(aux, label, bins=32):
    """
    Unpads 1D auxiliary outputs
    
    Input: 
        aux  : tuple of auxiliary outputs - result of predict crops, 2nd item
        label: torch tensor of shape (k, 70, 64)
    
    Output:
        unpadded_dict: dictionary of unpadded crops for each auxiliary output
    """
    unpadded_dict = {'sec_i':[], 'sec_j':[], 'phi_i':[], 
                     'phi_j':[], 'psi_i':[], 'psi_j':[]}
      
    for i in range(aux[0].shape[0]):
        for j, k in enumerate(unpadded_dict):
            nonzero_ind = np.nonzero(label[i, 64+j, :]).t()[0]
            unpadded_dict[k].append(aux[j][i, :, :, nonzero_ind].unsqueeze(0))

    return unpadded_dict


def aux_indices(k0, axis):
    """
    Returns indices that compose either "i" or "j" auxiliary output
    
    Input:
        k0  : number of crops per domain
        axis: char, either "i" or "j"
        
    Output:
        aux_ind: list of indices
    """
    
    k = int(np.sqrt(k0))
    indices = np.arange(k ** 2).reshape(k, k)
    aux_ind = []
    
    i0 = 0
    
    if axis == 'i':
        for i in range(k):
            aux_ind.append(indices[:, i])
    elif axis == 'j':
        for i in range(k):
            aux_ind.append(indices[i])
    return aux_ind


# %%
def unpad_and_glue_aux(unpadded_dict, aux_key, channels, L):
    """
    Glue auxiliary outputs separately by type and dimension
    
    Input:
        unpadded_dict: result of "unpad_aux" function
        aux_key      : string, auxiliary output type (eg 'sec_i')
        channels     : int, number of channels
        L            : int, domain length
        
    Output:
        aux_out: torch tensor of glued aux output. This output is also on the 
                 right scale (exp op was used)
    """
    temp_aux = unpadded_dict[aux_key]
    
    aux_out = torch.empty((8, channels, 1, L))
    
    if L % 64 == 0:
        ind_start = 0
        for i in range(8):
            k_temp = (L // 64 + (1 - i % 2)) ** 2  # k if even, k0 if odd
            
            if aux_key[4] == 'i':
                aux_ind = aux_indices(k_temp, 'i')
            else:
                aux_ind = aux_indices(k_temp, 'j')
            
            temp = temp_aux[ind_start:(ind_start + k_temp)]
            temp_tensor = torch.empty((int(np.sqrt(k_temp)), channels, 1, L))
            
            for ind, auind in enumerate(aux_ind):
                toglue = []
                for ai in auind:
                    toglue.append(torch.exp(temp[ai]))
                temp_tensor[ind] = torch.cat(toglue, dim=3)
            
            aux_out[i] = torch.mean(temp_tensor, dim=0)
            ind_start += k_temp
    
    else:
        k = (L // 64 + 1) ** 2
        ind_start = 0
        for i in range(8):
            if aux_key[4] == 'i':
                aux_ind = aux_indices(k, 'i')
            else:
                aux_ind = aux_indices(k, 'j')
            
            temp = temp_aux[ind_start:(ind_start + k)]
            temp_tensor = torch.empty((int(np.sqrt(k)), channels, 1, L))
            
            for ind, auind in enumerate(aux_ind):
                toglue = []
                for ai in auind:
                    toglue.append(torch.exp(temp[ai]))
                temp_tensor[ind] = torch.cat(toglue, dim=3)
            
            aux_out[i] = torch.mean(temp_tensor, dim=0)
            ind_start += k
    
    return torch.mean(aux_out, dim=0).unsqueeze(0)
           

#%%
def prepare_aux(aux, label, L):
    """
    Wrapper function for unpadding and gluing aux outputs
    
    Input:
        aux  : tuple of auxiliary outputs - result of predict crops, 2nd item
        label: torch tensor of shape (k, 70, 64)
        L    : int, domain length
        
    Output:
        aux_dict: dictionary of predicted, unpadded, glued, averaged auxiliary
                  outputs ['sec', 'phi', 'psi']
    """
    
    unpadded_dict = unpad_aux(aux, label)
    aux_keys = list(unpadded_dict.keys())
    aux_dict = {}

    for i in range(3):
        if aux_keys[2*i].startswith('sec'):
            channels = 9
        else:
            channels = 37
        
        aux_i = unpad_and_glue_aux(unpadded_dict, aux_keys[i * 2], channels, L)
        aux_j = unpad_and_glue_aux(unpadded_dict, aux_keys[i * 2 + 1], channels, L) 
        
        aux_dict[aux_keys[i * 2][:3]] = (aux_i + aux_j) / 2
    return aux_dict
    

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


def predict_outputs(model, domain, bins=32, device="cpu"):
    """
    Wrapper function for predicting a probability distribution of distances (distogram)
    
    Input:
        model : neural network with method "predict"
        domain: string
        bins  : int (32 is default), number of bins in resulting distograms
        device: ("cpu" - default or "cuda")
    
    Output:
        outputs_dict: dictionary of predicted, unpadded, glued, averaged auxiliary
                  outputs ['sec', 'phi', 'psi'] and distogram ['distogram']
    """
    
    L = domain_lengths[domain]
    
    distograms = torch.empty((8, 32, L, L))
    
    i, o = make_crops(domain)
    p_dist, p_aux = predict_crops(model, i, bins, device)
    
    # DISTOGRAM
    k = (L // 64 + 1) ** 2
    if L % 64 == 0:
        #k0 = (L // 64) ** 2
        s = 0
        for i in range(8):
            k_temp = (L // 64 + (1 - i % 2)) ** 2  # k if even, k0 if odd
            distograms[i] = unpad_and_glue(p_dist[s:(s+k_temp)], o[s:(s+k_temp), :64, :], L)
            s += k_temp
    else:
        for i in range(8):
            distograms[i] = unpad_and_glue(p_dist[(i * k):((i + 1) * k)], o[(i * k):((i + 1) * k), :64, :], L)
    
    distogram = torch.mean(distograms, dim=0)
    
    # AUXILIARY OUTPUTS
    outputs_dict = prepare_aux(p_aux, o, L)
    
    # add distogram to the dictionary
    outputs_dict['distogram'] = distogram
    
    return outputs_dict



