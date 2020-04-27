#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:32:26 2020

@author: andyb
"""

# %%
from model_aux import AlphaFold
import torch
import yaml
import argparse


# %%
def average_predictions(model, X, cfg):

    def initialize(shape):
        return torch.zeros(shape), torch.zeros(shape)

    def update(predictions, counts, location, values):
        predictions[location] += values.cpu()
        counts[location] += 1
        return None

    B, C, H, W = X.shape

    pred_dist, n_dist = initialize((B, cfg['dist_channels'], H, W))
    pred_ss, n_ss = initialize((B, cfg['ss_channels'], H))
    pred_phi, n_phi = initialize((B, cfg['phi_channels'], H))
    pred_psi, n_psi = initialize((B, cfg['psi_channels'], H))

    crop_size = cfg['crop_size']
    full = slice(0, None)
    for i in range(0, H - crop_size + 1):
        for j in range(0, W - crop_size + 1):
            i_slice = slice(i, i + crop_size)
            j_slice = slice(j, j + crop_size)

            loc = (full, full, i_slice, j_slice)

            with torch.no_grad():
                model.eval()
                dist, ss, phi, psi = model.predict(X[loc])

            update(pred_dist, n_dist, loc, dist)

            loc = (full, full, i_slice)
            update(pred_ss, n_ss, loc, ss)
            update(pred_phi, n_phi, loc, phi)
            update(pred_psi, n_psi, loc, psi)

    return (
        pred_dist / n_dist,
        pred_ss / n_ss,
        pred_phi / n_phi,
        pred_psi / n_psi
    )


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True)
parser.add_argument('--model_file', required=True)
parser.add_argument('--input_file', required=True)
parser.add_argument('--output_file', required=True)
args = parser.parse_args()

# %%
# args = argparse.Namespace()
# args.config_file = \
#     "/faststorage/project/deeply_thinking_potato/data/our_input/" \
#     "model_270420/alphafold_aux.yml"

# args.model_file = \
#     "/faststorage/project/deeply_thinking_potato/data/our_input/" \
#     "model_270420/0.pt"

# args.input_file = \
#     "/faststorage/project/deeply_thinking_potato/data/our_input/" \
#     "tensors/1bkbA02_X.pt"

# args.output_file = \
#     "/faststorage/project/deeply_thinking_potato/data/our_input/" \
#     "tensors_Y_pred/1bkbA02_X.pt"

# %%
cfg = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

# %%
if cfg['use_cuda'] and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %%
model = AlphaFold(
    in_channels=cfg['in_channels'],
    dist_channels=cfg['dist_channels'],
    ss_channels=cfg['ss_channels'],
    phi_channels=cfg['phi_channels'],
    psi_channels=cfg['psi_channels'],
    up_channels=cfg['up_channels'],
    down_channels=cfg['down_channels'],
    RESNET_depth=cfg['RESNET_depth'],
    crop_size=cfg['crop_size'],
    weights=cfg['weights']
)

model.to(device=device)
model.load_state_dict(torch.load(args.model_file, map_location=device))
model.eval()

# %%
X = (torch.load(args.input_file)
     .to(dtype=torch.float32, device=device)
     .unsqueeze(0))

# %%
Y_pred = average_predictions(model, X, cfg)

# %%
torch.save(Y_pred, args.output_file)
