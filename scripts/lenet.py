'''
LeNet
'''

#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import itertools
import glob
import time

from input_preparation import open_data
#%% LeNet
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.conv3 = nn.Conv2d(64, 86, 3, padding = 2)
        
        self.avgpool = nn.AvgPool2d(4)
        
        #self.linear1 = nn.Linear(16 * 4 * 4 + 21 + 21, 128)
        self.linear2 = nn.Linear(128, 64)

    def forward(self, x):
        J, s1, s2 = x
        J = J.view(1, 1, 22, 22)
        J = F.max_pool2d(torch.relu(self.conv1(J)), 2)  # 22 -> 20 -> 10
        J = F.max_pool2d(torch.relu(self.conv2(J)), 2)  # 10 -> 8 -> 4
        J = torch.relu(self.conv3(J))
        J = self.avgpool(J)
        J = J.view(-1, 86)                      # flatten
        s1 = s1.view(1, -1)
        s2 = s2.view(1, -1)
        x = torch.cat((J, s1, s2), dim=1)
        #x = torch.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x), 1)
        return x

    def fit(self, X, Y, loss_function, optimizer):
        self.train()

        L = Y.shape[0]
        #indices = list(itertools.product(range(L), range(L)))
        #indices = np.random.permutation(indices)
        indices = np.array([np.random.choice(np.arange(L), size=L**2), 
                            np.random.choice(np.arange(L), size=L**2)]).T
        
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

            #loss = loss_function(predicted, Y[index[0], index[1]].view(1, -1).argmax(dim=1))
            loss = F.nll_loss(predicted, Y[index[0], index[1]].view(1).long())
            
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



#%%
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
