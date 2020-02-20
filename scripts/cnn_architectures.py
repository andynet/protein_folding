'''
File contains several CNN architectures (LeNet, AlexNet, GoogleNet, ...)
that are tailored to work with our input.

For the initial try the input is only 22x22 matrix J(i, j) on which the
convolution operations are performed. In the flattening step a one hot encoded
aminoacid positions are added.
'''

# %% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import itertools
import glob
import time


# %% LeNet
class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.linear1 = nn.Linear(16 * 4 * 4 + 21 + 21, 128)
        self.linear2 = nn.Linear(128, 64)

    def forward(self, x):
        J, s1, s2 = x
        J = J.view(1, 1, 22, 22)
        J = F.max_pool2d(torch.relu(self.conv1(J)), 2)  # 22 -> 20 -> 10
        J = F.max_pool2d(torch.relu(self.conv2(J)), 2)  # 10 -> 8 -> 4
        J = J.view(-1, 16 * 4 * 4)                      # flatten
        s1 = s1.view(1, -1)
        s2 = s2.view(1, -1)
        x = torch.cat((J, s1, s2), dim=1)
        x = torch.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x), 1)
        return x

    def fit(self, X, Y, loss_function, optimizer):
        self.train()

        L = Y.shape[0]
        indices = list(itertools.product(range(L), range(L)))
        indices = np.random.permutation(indices)
        for i, index in enumerate(indices):
            if i % 1000 == 0:
                print(i, '/', len(indices))
            # if i > 25000:
            #     break
            # J = X[0][index[0], index[1]]
            # s1 = X[1][index[0]]
            # s2 = X[1][index[1]]
            # y = Y[index[0], index[1]]

            # x = (X[0][index[0], index[1]], X[1][index[0]], X[1][index[1]])

            optimizer.zero_grad()
            # print(time.time())
            predicted = self.forward((X[0][index[0], index[1]],
                                      X[1][index[0]], X[1][index[1]]))
            # print(time.time())

            loss = loss_function(predicted, Y[index[0], index[1]].view(1, -1).argmax(dim=1))
            loss.backward()
            optimizer.step()

    def predict(self, x):
        return self.forward(x)

    def score(self, X, Y, loss_function):
        self.eval()

        L = Y.shape[0]
        loss_total = 0
        indices = list(itertools.product(range(L), range(L)))
        for i, index in enumerate(indices):
            if i % 1000 == 0:
                print(i, '/', len(indices))
            # if i > 25000:
            #     break
            J = X[0][index[0], index[1]]
            s1 = X[1][index[0]]
            s2 = X[1][index[1]]
            y = Y[index[0], index[1]]
            x = (J, s1, s2)
            predicted = self.forward(x)
            loss_total += loss_function(predicted, y.view(1, -1).argmax(dim=1))

        return float(loss_total)


# %%
def open_pickle(filepath):
    '''Opens ProSPr pickle file and extracts the content'''
    objects = []
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return np.array(objects)


# %%
def seq_to_dummy_tensor(seq):
    order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
             'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']

    a = np.array([aa for aa in seq])
    b = pd.get_dummies(a)
    b = b.loc[:, order]

    return b.values


# %%
def output_to_dummy_tensor(output):
    L = output.shape[0]
    result = np.zeros((L, L, 64))
    for i in range(L):
        for j in range(L):
            k = output[i, j]
            result[i, j, k] = 1

    return result


# %%
def open_data(domains):
    tensors = {}
    for domain in domains:
        filepath = f'../data/potts/{domain}.pkl'
        tensors[domain] = open_pickle(filepath)[0]

    sequences = open_pickle('../data/ProSPr/name2seq.pkl')[0]
    seqs = {key: value for key, value in sequences.items() if key in domains}

    for domain in domains:
        tensors[domain]['seq'] = seq_to_dummy_tensor(seqs[domain])

    output = open_pickle('../data/ProSPr/name2bins.pkl')[0]
    for domain in domains:
        tensors[domain]['output'] = output_to_dummy_tensor(output[domain])

    return tensors


# %% LeNet trial
# J_ln = potts_input('5inwA02', 20, 35)
# J_ln = J_ln.view(1, 1, 22, 22)
# s1_ln, s2_ln = ohe_seq_ij('5inwA02', 20, 35)
#
# lnmod = lenet()
# print(lnmod((J_ln, s1_ln, s2_ln)))

# %% Training Algorithm
#
# train_domains = ['5ffdA00', '8abpA01', '5l73A00', '5ib0A00', '5eyaF00', '5jicA02']
# val_domains = ['5eyfA02', '5h02A02', '5jlaA00']
#
# Someday when all pickles are downloaded
# train_domains = open_pickle('../data/ProSPr/TRAIN-p_names.pkl')
# validation_domains = open_pickle('../data/ProSPr/VS-p_names.pkl')
#

# %%
# train_domains = ['4j0cA00', '4j3vA03', '4j6cA00', '4j6oA00',
#                  '4j7aA00', '4j7qA00', '4j7rB01', '4j80A03']
# val_domains = ['4j8eA00', '4j8lA02']
# tmp = glob.glob('../data/potts/*.pkl')
# domains = [x.split('/')[-1].split('.')[0] for x in tmp]

train_domains = ['4j0cA00']
val_domains = ['4j0cA00']
message = 'epoch: {}\ttrain_loss: {}\tvalidation_loss: {}'

train_data = open_data(train_domains)
val_data = open_data(val_domains)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'device': device, 'dtype': torch.float32}
print(device)

model = lenet().to(**kwargs)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss = nn.NLLLoss()

train_hist = []
val_hist = []


# %%
for epoch in range(1, 5):
    for domain in train_domains:
        # inputs
        # print(time.time())
        x = (torch.tensor(train_data[domain]['J']).to(**kwargs),
             torch.tensor(train_data[domain]['seq']).to(**kwargs))
        # outputs
        # print(time.time())
        y = torch.tensor(train_data[domain]['output']).to(**kwargs)

        print(f'Training on domain {domain} - {epoch}. epoch.')
        print(time.time())
        model.fit(x, y, loss, optimizer)
        print(f'Done.')

    with torch.no_grad():
        train_loss = 0
        for domain in train_domains:
            # inputs
            x = (torch.tensor(train_data[domain]['J']).to(**kwargs),
                 torch.tensor(train_data[domain]['seq']).to(**kwargs))
            # outputs
            y = torch.tensor(train_data[domain]['output']).to(**kwargs)

            train_loss += model.score(x, y, loss)

        train_loss /= len(train_domains)
        train_hist.append(train_loss)

        validation_loss = 0
        for domain in val_domains:
            # inputs
            x = (torch.tensor(train_data[domain]['J']).to(**kwargs),
                 torch.tensor(train_data[domain]['seq']).to(**kwargs))
            # outputs
            y = torch.tensor(train_data[domain]['output']).to(**kwargs)

            validation_loss += model.score(x, y, loss)

        validation_loss /= len(val_domains)
        val_hist.append(validation_loss)

        print(message.format(epoch,
                             round(train_loss, 2),
                             round(validation_loss, 2)))

# %%
torch.save(model.state_dict(), 'model.pth')
