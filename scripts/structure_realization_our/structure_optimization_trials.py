#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Picking good hyperparameters for optimization
"""
# %%
from optimize import optimize
import pickle
import matplotlib.pyplot as plt

# SMALL DOMAIN

# %% Learning rate
lr0 = optimize('1a02F00', iterations=200, lr=1, random_state=1618)
lr1 = optimize('1a02F00', iterations=200, lr=1e-1, random_state=1618)
lr2 = optimize('1a02F00', iterations=200, lr=1e-2, random_state=1618)
lr3 = optimize('1a02F00', iterations=200, lr=1e-3, random_state=1618)
lr4 = optimize('1a02F00', iterations=200, lr=1e-4, random_state=1618)

# %%
small_lr = {'lr0':lr0, 'lr1':lr1, 'lr2':lr2, 'lr3':lr3, 'lr4':lr4}

with open('../../steps/structure_realization_trials/small_lr.pkl', 'wb') as f:
    pickle.dump(small_lr, f)

# %%
#small_lr['lr0'][0].visualize_structure()

for i in range(4):
    plt.plot(small_lr[f'lr{i}'][2][:, 0], small_lr[f'lr{i}'][2][:, 1], label=f'lr=1e-{i}')

plt.legend()
plt.title('Effect of Learning Rate')

plt.savefig('../../plots/structure_realization/small_lr.png')

# %% Learning rate decay
lrd0 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.9, random_state=1618)
lrd1 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.8, random_state=1618)
lrd2 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.7, random_state=1618)

# %%
small_lrd = {'lrd0':lrd0, 'lrd1':lrd1, 'lrd2':lrd2}

with open('../../steps/structure_realization_trials/small_lrd.pkl', 'wb') as f:
    pickle.dump(small_lrd, f)

# %%
plt.plot(small_lrd['lrd0'][2][:, 0], small_lrd['lrd0'][2][:, 1], label=f'lr=1, lr_decay=0.9')
plt.plot(small_lrd['lrd1'][2][:, 0], small_lrd['lrd1'][2][:, 1], label=f'lr=1, lr_decay=0.8')
plt.plot(small_lrd['lrd2'][2][:, 0], small_lrd['lrd2'][2][:, 1], label=f'lr=1, lr_decay=0.7')

plt.legend()
plt.title('Effect of Learning Rate Decay')
plt.savefig('../../plots/structure_realization/small_lr_decay.png')
# %% Momentum
m0 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.7, momentum=0.1, random_state=1618)
m1 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.7, momentum=0.3, random_state=1618)
m2 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.7, momentum=0.5, random_state=1618)

# %%
small_momentum = {'m0':m0, 'm1':m1, 'm2':m2}
with open('../../steps/structure_realization_trials/small_momentum.pkl', 'wb') as f:
    pickle.dump(small_momentum, f)

# %%
plt.plot(small_momentum['m0'][2][:, 0], small_momentum['m0'][2][:, 1], label=f'lr=1, lr_decay=0.7, momentum=0.1')
plt.plot(small_momentum['m1'][2][:, 0], small_momentum['m1'][2][:, 1], label=f'lr=1, lr_decay=0.7, momentum=0.3')
plt.plot(small_momentum['m2'][2][:, 0], small_momentum['m2'][2][:, 1], label=f'lr=1, lr_decay=0.7, momentum=0.5')

plt.legend()
plt.title('Effect of Momentum')
plt.savefig('../../plots/structure_realization/small_lr_momentum.png')

# %% Nesterov Momentum
n0 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.7, momentum=0.1, nesterov=True, random_state=1618)
n1 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.7, momentum=0.3, nesterov=True, random_state=1618)
n2 = optimize('1a02F00', iterations=200, lr=1, lr_decay=0.7, momentum=0.5, nesterov=True, random_state=1618)

# %%
small_nesterov = {'n0':n0, 'n1':n1, 'n2':n2}
with open('../../steps/structure_realization_trials/small_nesterov.pkl', 'wb') as f:
    pickle.dump(small_nesterov, f)

# %%
plt.plot(small_nesterov['n0'][2][:, 0], small_nesterov['n0'][2][:, 1], label=f'lr=1, lr_decay=0.7, nesterov=0.1')
plt.plot(small_nesterov['n1'][2][:, 0], small_nesterov['n1'][2][:, 1], label=f'lr=1, lr_decay=0.7, nesterov=0.3')
plt.plot(small_nesterov['n2'][2][:, 0], small_nesterov['n2'][2][:, 1], label=f'lr=1, lr_decay=0.7, nesterov=0.5')

plt.legend()
plt.title('Effect of Nesterov Momentum')
plt.savefig('../../plots/structure_realization/small_lr_nesterov.png')
# %%
with open('../../steps/structure_realization_trials/small_lr.pkl', 'rb') as f:
    small_lr = pickle.load(f)
    
with open('../../steps/structure_realization_trials/small_lrd.pkl', 'rb') as f:
    small_lrd = pickle.load(f)

# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 12))

# learning rate
for i in range(4):
    ax[0, 0].plot(small_lr[f'lr{i}'][2][:, 0], small_lr[f'lr{i}'][2][:, 1], label=f'lr=1e-{i}')

ax[0, 0].set_title('Effect of Learning Rate')

# learning rate decay
ax[0, 1].plot(small_lrd['lrd0'][2][:, 0], small_lrd['lrd0'][2][:, 1], label=f'lr=1, lr_decay=0.9')
ax[0, 1].plot(small_lrd['lrd1'][2][:, 0], small_lrd['lrd1'][2][:, 1], label=f'lr=1, lr_decay=0.8')
ax[0, 1].plot(small_lrd['lrd2'][2][:, 0], small_lrd['lrd2'][2][:, 1], label=f'lr=1, lr_decay=0.7')
ax[0, 1].set_title('Effect of Learning Rate Decay')

# Momentum
ax[1, 0].plot(small_momentum['m0'][2][:, 0], small_momentum['m0'][2][:, 1], label=f'lr=1, lr_decay=0.7, momentum=0.1')
ax[1, 0].plot(small_momentum['m1'][2][:, 0], small_momentum['m1'][2][:, 1], label=f'lr=1, lr_decay=0.7, momentum=0.3')
ax[1, 0].plot(small_momentum['m2'][2][:, 0], small_momentum['m2'][2][:, 1], label=f'lr=1, lr_decay=0.7, momentum=0.5')
ax[1, 0].set_title('Effect of Momentum')

# Nesterov Momentum
ax[1, 1].plot(small_nesterov['n0'][2][:, 0], small_nesterov['n0'][2][:, 1], label=f'lr=1, lr_decay=0.7, nesterov=0.1')
ax[1, 1].plot(small_nesterov['n1'][2][:, 0], small_nesterov['n1'][2][:, 1], label=f'lr=1, lr_decay=0.7, nesterov=0.3')
ax[1, 1].plot(small_nesterov['n2'][2][:, 0], small_nesterov['n2'][2][:, 1], label=f'lr=1, lr_decay=0.7, nesterov=0.5')
ax[1, 1].set_title('Effect of Nesterov Momentum')

for i in range(2):
    for j in range(2):
        ax[i, j].legend()
        ax[i, j].set_ylim(2000, 9500)

fig.suptitle('Optimization Hyperparameters', fontsize=20)
plt.savefig('../../plots/structure_realization/optimization_hyperparameters.png', dpi=200)