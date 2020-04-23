#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:32:26 2020

@author: andyb
"""

# %%
from model_aux import AlphaFold, Domains
import torch
import yaml
import matplotlib.pyplot as plt

# %%
config_file = "alphafold_aux.yml"
epoch = '8'
domains = [
    # train
    '1bkbA02', '1bsmA01', '1c8cA00', '1cukA01', '1d5yA02',
    '16pkA01', '1a04A01', '1a2oA01', '1a2pA00', '1a41A02',
    # validation
    '1wwuA01', '1wxuA01', '1x6bA01', '1yg0A00', '1yhnB00',
    '1ynjJ03', '1zud400', '1whlA00', '1whmA01', '1whqA01',
]

# %%
cfg = yaml.load(open(config_file), Loader=yaml.FullLoader)

# %%
if cfg['use_cuda'] and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %%
model = AlphaFold(
    input_size=cfg['input_size'],
    up_size=cfg['up_size'],
    down_size=cfg['down_size'],
    output_size=cfg['output_size'],
    aux_size=cfg['aux_size'],
    crop_size=cfg['crop_size'],
    RESNET_depth=cfg['RESNET_depth']
).to(device=device)

model.load_state_dict(torch.load(cfg['model_path'].format(epoch=epoch)))
model.eval()


# %%
dataset = Domains(domains[0:1], cfg['X_files'], cfg['Y_files'], verbosity=0)
loader = torch.utils.data.DataLoader(dataset, num_workers=cfg['num_workers'])

# %%
loaded_data = []
for i, (X, Y, name) in enumerate(loader):
    length = X.shape[2]
    if length >= cfg['crop_size']:
        loaded_data.append((X, Y, name, length))


# %%
# def predict(model, data, cfg, device):
#     X, Y, name, length = data
#     X = X.to(device=device)

#     dist_predicted = torch.zeros((1, cfg['output_size'], length, length))
#     dist_count = torch.zeros((1, cfg['output_size'], length, length))
#     ones = torch.ones((1, cfg['output_size'], cfg['crop_size'], cfg['crop_size']))

#     for i in range(0, length - cfg['crop_size'] + 1):
#         for j in range(0, length - cfg['crop_size'] + 1):
#             i_slice = slice(i, i + cfg['crop_size'])
#             j_slice = slice(j, j + cfg['crop_size'])

#             prediction = model.predict(X[:, :, i_slice, j_slice])
#             dist, ss, phi, psi = prediction

#             dist_predicted[:, :, i_slice, j_slice] += dist.cpu().detach()
#             dist_count[:, :, i_slice, j_slice] += ones

#     result = dist_predicted / dist_count
#     return result
# %%
def predict(model, X):
    """
    X.shape = (B, C, H, W)
    """
    model.eval()
    B, C, H, W = X.shape

    dist_predicted = torch.zeros((B, model.output_size, H, W))
    dist_count = torch.zeros((B, C, H, W))
    ones = torch.ones((B, C, model.crop_size, model.crop_size))

    ss_predicted = torch.zeros((B, C, H))
    ss_count = torch.zeros((B, C, H))
    ones2 = torch.ones((B, C, model.crop_size))

    for i in range(0, H - model.crop_size + 1):
        for j in range(0, W - model.crop_size + 1):
            i_slice = slice(i, i + model.crop_size)
            j_slice = slice(j, j + model.crop_size)

            prediction = model.forward(X[:, :, i_slice, j_slice])
            dist, ss, phi, psi = prediction

            dist_predicted[:, :, i_slice, j_slice] += dist
            dist_count[:, :, i_slice, j_slice] += ones

            ss_predicted[:, :, i_slice] += ss
            ss_count[:, :, i_slice] += ones2

    dist_predicted /= dist_count
    ss_predicted /= ss_count
    return dist_predicted, ss_predicted


# %%
with torch.no_grad():
    pred = model.predict(loaded_data[0][0][:, :, 0:100, 0:100].to(device=device))

# %%
# result = predict(model, loaded_data[0], cfg, device)
dist = pred[0].squeeze().argmax(dim=0)
aux = pred[1].squeeze()


# %%
def visualize(Y, prediction):
    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False)

    # ax[0, 0].imshow(Y, cmap='viridis_r')
    # ax[0, 0].set_title("Y")

    # ax[0, 1].bar(x=list(Y_count.keys()), height=list(Y_count.values()))
    # ax[0, 1].set_title("Y_count")

    ax[0, 0].imshow(Y, cmap='viridis_r')
    ax[0, 0].set_title("Y")

    ax[0, 1].imshow(prediction, cmap='viridis_r')
    ax[0, 1].set_title("prediction")

    plt.tight_layout()
    plt.show()


visualize(Y[0].squeeze().numpy()[0:100, 0:100], dist.numpy())


# %%
def create_dummy(tensor, n_cat):
    tensor = tensor.transpose(1, 0)
    one_hot = torch.zeros(tensor.shape[0], n_cat)
    one_hot = one_hot.scatter(1, tensor, 1)
    return one_hot.transpose(1, 0)


# %%
ss_dummy = create_dummy(Y[1], 8)
phi_dummy = create_dummy(Y[2], 37)
psi_dummy = create_dummy(Y[3], 37)

visualize(torch.cat([ss_dummy, phi_dummy, psi_dummy], dim=0).numpy()[:, 0:50], aux.numpy()[:, 0:50])
