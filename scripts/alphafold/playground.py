#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:29:11 2020

@author: andyb
"""

# %%
import torch
import matplotlib.pyplot as plt

# %%
pred_Y_file = "/faststorage/project/deeply_thinking_potato/data/our_input/tensors_Y_pred/1bkbA02_Y.pt"
real_Y_file = "/faststorage/project/deeply_thinking_potato/data/our_input/Y_tensors/1bkbA02_Y.pt"

pred_Y = torch.load(pred_Y_file)
real_Y = torch.load(real_Y_file)


# %%
def create_dummy(tensor, n_cat):
    tensor = tensor.transpose(1, 0)
    one_hot = torch.zeros(tensor.shape[0], n_cat)
    one_hot = one_hot.scatter(1, tensor, 1)
    return one_hot.transpose(1, 0)


# %%
def visualize(real_Y, pred_Y):
    fig, ax = plt.subplots(nrows=4, ncols=2, squeeze=False)

    ax[0, 0].imshow(real_Y[0], cmap='viridis_r')
    ax[0, 0].set_title("real_Y")

    ax[0, 1].imshow(pred_Y[0].argmax(dim=1).squeeze(), cmap='viridis_r')
    ax[0, 1].set_title("pred_Y")

    ss_dummy = create_dummy(real_Y[1].unsqueeze(0).to(dtype=torch.int64), 8)
    ax[1, 0].imshow(ss_dummy, cmap='viridis_r')
    ax[1, 0].set_title("real_Y")

    ax[1, 1].imshow(pred_Y[1].squeeze(), cmap='viridis_r')
    ax[1, 1].set_title("pred_Y")

    phi_dummy = create_dummy(real_Y[2].unsqueeze(0).to(dtype=torch.int64), 37)
    ax[2, 0].imshow(phi_dummy, cmap='viridis_r')
    ax[2, 0].set_title("real_Y")

    ax[2, 1].imshow(pred_Y[2].squeeze(), cmap='viridis_r')
    ax[2, 1].set_title("pred_Y")

    psi_dummy = create_dummy(real_Y[3].unsqueeze(0).to(dtype=torch.int64), 37)
    ax[3, 0].imshow(psi_dummy, cmap='viridis_r')
    ax[3, 0].set_title("real_Y")

    ax[3, 1].imshow(pred_Y[3].squeeze(), cmap='viridis_r')
    ax[3, 1].set_title("pred_Y")

    plt.tight_layout()
    plt.show()


visualize(real_Y, pred_Y)

# %%
criterion = torch.nn.CrossEntropyLoss()

loss = (
    10 * criterion(pred_Y[0], real_Y[0].unsqueeze(0))
    + 2 * criterion(pred_Y[1], real_Y[1].unsqueeze(0).to(dtype=torch.int64))
    + 2 * criterion(pred_Y[2], real_Y[2].unsqueeze(0))
    + 2 * criterion(pred_Y[3], real_Y[3].unsqueeze(0))
)

# %% testing?
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['l2']))


# B = 1
# dist_pred = torch.ones((B, 64, 64)).to(dtype=torch.int64)
# ss_pred = torch.ones((B, 64)).to(dtype=torch.int64)
# phi_pred = torch.ones((B, 64)).to(dtype=torch.int64)
# psi_pred = torch.ones((B, 64)).to(dtype=torch.int64)
# Y = (dist_pred, ss_pred, phi_pred, psi_pred)

# # %%
# with torch.no_grad():
#     model.eval()
#     pred = model.predict(X[:, :, 0:64, 0:64])
#     loss = model.score(X[:, :, 0:64, 0:64], Y, criterion)
#     print(loss)

# # %%
# model.train()
# for i in range(10):
#     model.fit(X[:, :, 0:64, 0:64], Y, criterion, optimizer)
# %%
# criterion = torch.nn.CrossEntropyLoss()

# B = 1
# dist_pred = torch.ones((B, 32, 64, 64))
# ss_pred = torch.ones((B, 8, 64))
# phi_pred = torch.ones((B, 37, 64))
# psi_pred = torch.ones((B, 37, 64))

# loss = 10 * criterion(dist_pred, dist[:, 0:64, 0:64])
# loss += 2 * criterion(ss_pred, ss[:, 0:64])
# loss += 2 * criterion(phi_pred, phi[:, 0:64])
# loss += 2 * criterion(psi_pred, psi[:, 0:64])

# print(loss)
