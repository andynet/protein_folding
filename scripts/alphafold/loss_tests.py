#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:15:26 2020

@author: andyb
"""

# %%
import matplotlib.pyplot as plt
from collections import Counter
from models import AlphaFold
import numpy as np
import torch


# %%
def visualize(Y, prediction):
    Y = np.array(Y).squeeze()
    Y_count = Counter(sorted(Y.flatten()))

    prediction = np.array(prediction.argmax(dim=1)).squeeze()
    prediction_count = Counter(sorted(prediction.flatten()))

    fig, ax = plt.subplots(nrows=2, ncols=2)

    # ax[0, 0].imshow(Y, cmap='viridis_r')
    # ax[0, 0].set_title("Y")

    # ax[0, 1].bar(x=list(Y_count.keys()), height=list(Y_count.values()))
    # ax[0, 1].set_title("Y_count")

    ax[1, 0].imshow(prediction, cmap='viridis_r')
    ax[1, 0].set_title("prediction")

    ax[1, 1].bar(x=list(prediction_count.keys()), height=list(prediction_count.values()))
    ax[1, 1].set_title("prediction_count")

    plt.tight_layout()
    plt.show()


# %%
batch_size = 1
crop_size = 32
RESNET_depth = 192

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = 0
torch.manual_seed(seed)
np.random.seed(seed=seed)
torch.cuda.manual_seed_all(seed)

# %%
model = AlphaFold(input_size=675, up_size=128, down_size=64, output_size=64,
                  RESNET_depth=RESNET_depth).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = torch.nn.CrossEntropyLoss()

# %%
# random input, random output, random prediction
X = torch.rand((batch_size, 675, crop_size, crop_size))
Y = torch.randint(64, (batch_size, crop_size, crop_size))
# multiplying predictions by constant increases loss, why?
prediction = torch.rand((batch_size, 64, crop_size, crop_size)) * 100
print(round(criterion(prediction, Y).item(), 4))
visualize(Y, prediction)

# %%
# random input, random output, real prediction
X = torch.rand((batch_size, 675, crop_size, crop_size))
Y = torch.randint(64, (batch_size, crop_size, crop_size))
prediction = model.predict(X.to(dtype=torch.float32))
print(round(criterion(prediction, Y).item(), 4))
visualize(Y, prediction)

# %%
# real input, real output, random prediction
i, j = 0, 0
X, Y = torch.load('/faststorage/project/deeply_thinking_potato/data/prospr/tensors_cs64/1cl8A00.pt')
X = X.view(1, 675, 261, 261)[:, :, i:i + crop_size, j:j + crop_size]
Y = Y.view(1, 261, 261)[:, i:i + crop_size, j:j + crop_size]
prediction = torch.rand((batch_size, 64, crop_size, crop_size))
print(round(criterion(prediction, Y).item(), 4))
visualize(Y, prediction)

# %%
# real input, real output, real prediction
i, j = 0, 0
X, Y = torch.load('/faststorage/project/deeply_thinking_potato/data/prospr/tensors_cs64/1cl8A00.pt')

X = X.view(1, 675, 261, 261)[:, :, i:i + crop_size, j:j + crop_size]\
    .to(device=device, dtype=torch.float32)

Y = Y.view(1, 261, 261)[:, i:i + crop_size, j:j + crop_size]\
    .to(device=device)

prediction = model.predict(X)
print(round(criterion(prediction, Y).item(), 4))
visualize(Y.cpu(), prediction.cpu())

# %%
history = [model.score(X, Y, criterion).item()]

# %%
for i in range(200):
    model.fit(X, Y, criterion, optimizer)
    prediction = model.predict(X)
    loss = model.score(X, Y, criterion)
    print(round(loss.item(), 6))
    history.append(loss.item())
visualize(Y.cpu(), prediction.cpu())

# %%
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(range(len(history)), history)

# %%
# load saved model?
# store local extreme loss models
