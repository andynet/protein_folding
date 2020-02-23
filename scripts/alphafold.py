'''
AlphaFolds Architecture
'''

#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math as m

from input_preparation import open_pickle
#%% Constants
INPUT_LAYERS = 613
INPUT_DIMENSION = 32

AFBLOCK_1_INPUT_LAYERS = 256
AFBLOCK_1_INPUT_LOOPS = 1

AFBLOCK_2_INPUT_LAYERS = 128
AFBLOCK_2_INPUT_LOOPS = 1

AVGPOOL_DIM = 128
BINS = 64
CROP_SIZE = 32

EPOCHS = 1
EVAL_DOMAINS = 1

#%% AlphaFold class

class AFdilationblock(nn.Module):
    """ Dilation Block
    Creates a block described by the image in the paper with specific dilation    
    """
    
    def __init__(self, in_channels, dilation):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.projectdown = nn.Conv2d(in_channels, in_channels//2, 1)
        self.bn2 = nn.BatchNorm2d(in_channels//2)
        self.conv = nn.Conv2d(in_channels//2, in_channels//2, 3, padding= dilation, dilation = dilation)
        self.bn3 = nn.BatchNorm2d(in_channels//2)
        self.projectup = nn.Conv2d(in_channels//2, in_channels, 1)
        
    def forward(self, x):
        identity = x
        x = F.elu(self.bn1(x))
        x = self.projectdown(x)
        x = F.elu(self.bn2(x))
        x = self.conv(x)
        x = F.elu(self.bn3(x))
        x = self.projectup(x)
        return x + identity
    
class AFBlock(nn.Module):
    """Generates Set of dilation blocks with different dilation values"""
    
    def __init__(self, in_channels):
        super().__init__()
        dilation_block = AFdilationblock
        
        self.dilblock1 = dilation_block(in_channels, dilation = 1)
        self.dilblock2 = dilation_block(in_channels, dilation = 2)
        self.dilblock4 = dilation_block(in_channels, dilation = 4)
        self.dilblock8 = dilation_block(in_channels, dilation = 8)
        
    def forward(self, x):
        x = self.dilblock1(x)
        x = self.dilblock2(x)
        x = self.dilblock4(x)
        x = self.dilblock8(x)
        return x
    
class AlphaFold(nn.Module):
    
    def __init__(self):
        super().__init__()
        afblock = AFBlock
        
        # 1. Set the dimensionality of data set to INPUT_LAYERS
        self.conv1 = nn.Conv2d(INPUT_LAYERS, AFBLOCK_1_INPUT_LAYERS, 1)
        self.bnC1 = nn.BatchNorm2d(AFBLOCK_1_INPUT_LAYERS)
        
        # 2. First series of Convolutional-Dilation Blocks
        self.afblock1 = nn.ModuleList([afblock(AFBLOCK_1_INPUT_LAYERS) for i in range(AFBLOCK_1_INPUT_LOOPS)])
        self.bnAF1 = nn.BatchNorm2d(AFBLOCK_1_INPUT_LAYERS)
        
        # 3. Change dimensionality (AFB1_IL -> AFB2_IL)
        self.conv2 = nn.Conv2d(AFBLOCK_1_INPUT_LAYERS, AFBLOCK_2_INPUT_LAYERS, 1)
        self.bnC2 = nn.BatchNorm2d(AFBLOCK_2_INPUT_LAYERS)
        
        # 4. Second series of Convolutional-Dilation Blocks
        self.afblock2 = nn.ModuleList([afblock(AFBLOCK_2_INPUT_LAYERS) for i in range(AFBLOCK_2_INPUT_LOOPS)])
        self.bnAF2 = nn.BatchNorm2d(AFBLOCK_2_INPUT_LAYERS)
        
        # 5. Create output. I AM NOT SURE HOW TO END ALL THIS MADNESS. THIS IS ONE OPTION
        self.avgpool = nn.AvgPool2d(INPUT_DIMENSION)
        self.linear = nn.Linear(AFBLOCK_2_INPUT_LAYERS, BINS)
        
    def forward(self, x):
        
        # 1.
        x = self.conv1(x)
        x = F.elu(self.bnC1(x))
        
        # 2.
        for block in self.afblock1:
            x = block(x)
        x = self.bnAF1(x)
        
        # 3.
        x = self.conv2(x)
        x = self.bnC2(x)
        
        # 4.
        for block in self.afblock2:
            x = block(x)
        x = F.elu(self.bnAF2(x))
        
        # 5.
        x = self.avgpool(x)
        x = x.view(1, AFBLOCK_2_INPUT_LAYERS)
        x = F.log_softmax(self.linear(x), dim = 1)
        
        return x
    
    def fit(self, train_domains, validation_domains, optimizer):   
    
        for epoch in range(EPOCHS):
            
            # TRAINING
            self.train()
            for domain in train_domains:
                print('Training domain: ', domain)
                L = domain_lengths[domain]
                n_batches = ((L-1) // CROP_SIZE + 1)**2
                
                for b in range(n_batches):
                    batch_indices = np.array([np.random.choice(np.arange(L), size=CROP_SIZE**2), 
                                              np.random.choice(np.arange(L), size=CROP_SIZE**2)]).T
                    
                    # Here magic happens. There could be a function that loads
                    # input correspondent to the indices
                    batch_input = andyho_magia(domain, batch_indices) # = (input, output)
                    # input.shape = (CROP_SIZE**2, 1, INPUT_LAYERS, INPUT_DIMENSION, INPUT_DIMENSION)
                    # output.shape = (CROP_SIZE**2, 1)
                    
                    for i in range(CROP_SIZE**2):
                        prediction = self(batch_input[0][i])
                        loss = F.nll_loss(prediction, batch_input[1][i].view(1))
                        
                        loss.backward()
                        optimizer.step()        
            # TRAIN & VALIDATION ERROR
            # we can pick (randomly) few domains from train list and val domains and compute
            # the error random batch of indices 
            self.eval()
            
            # pick random indices of domains
            train_eval_domains_ind = np.random.choice(np.arange(len(train_domains)), EVAL_DOMAINS, replace = False)
            validation_eval_domains_ind = np.random.choice(np.arange(len(validation_domains)), EVAL_DOMAINS, replace = False)
            
            self.history = [] # Epoch, Train loss, Val loss
            train_losses, val_losses = [], []
            for i in range(EVAL_DOMAINS):
                
                # TRAIN ERROR
                domain = train_domains[train_eval_domains_ind[i]]
                L = domain_lengths[domain]
                batch_indices = np.array([np.random.choice(np.arange(L), size=CROP_SIZE**2), 
                                          np.random.choice(np.arange(L), size=CROP_SIZE**2)]).T
                temp_set = andyho_magia(domain, batch_indices)
                for c in range(CROP_SIZE**2):
                    temp_preds = self(temp_set[0][c])
                    train_losses.append(F.nll_loss(temp_preds, temp_set[1][c].view(1)).detach())
                
                # VALIDATION ERROR
                domain = validation_domains[validation_eval_domains_ind[i]]
                L = domain_lengths[domain]
                batch_indices = np.array([np.random.choice(np.arange(L), size=CROP_SIZE**2), 
                                          np.random.choice(np.arange(L), size=CROP_SIZE**2)]).T
                temp_set = andyho_magia(domain, batch_indices)
                for c in range(CROP_SIZE**2):
                    temp_preds = self(temp_set[0][c])
                    train_losses.append(F.nll_loss(temp_preds, temp_set[1][c].view(1)).detach())
                
            self.history.append([epoch, torch.mean(torch.tensor(train_losses)), torch.mean(torch.tensor(val_losses))])
            print(f'Epoch: {epoch}, Train_loss: {torch.mean(torch.tensor(train_losses))}, Validation loss: {torch.mean(torch.tensor(train_losses))}')

#%% Precompute Domain Lengths
sequences = open_pickle('../data/ProSPr/name2seq.pkl')[0]
domain_lengths = {}
for k in sequences.keys():
    domain_lengths[k] = len(sequences[k])
del k, sequences

#%% Crops indices
#def crops_indices(domain):
#    """Returns 2D list of crop indices of form:
#            [(i_min, j_min, i_max, j_max), ...]
#            where i_max and j_max indices should not be included in input
#    """
#    
#    domain_length = domain_lengths[domain]
#    ncrops = ((domain_length-1) // CROP_SIZE + 1) # just along one dimension
#    corners = [m.ceil(domain_length/ncrops) * i for i in range(ncrops)]
#    corners.append(domain_length)
#    
#    indices = []
#    
#    for i in range(len(corners)-1):
#        for j in range(len(corners)-1):
#            indices.append((corners[i], corners[j], corners[i+1], corners[j+1]))
#    return indices

#%% Simulate Input
        
INPUT_LAYERS = 5
INPUT_DIMENSION = 32

AFBLOCK_1_INPUT_LAYERS = 6
AFBLOCK_1_INPUT_LOOPS = 1

AFBLOCK_2_INPUT_LAYERS = 4
AFBLOCK_2_INPUT_LOOPS = 1

AVGPOOL_DIM = 4
BINS = 64
CROP_SIZE = 5

EPOCHS = 1
EVAL_DOMAINS = 1

def simulate_domain(L):
    return torch.rand((L, L, 1, INPUT_LAYERS, INPUT_DIMENSION, INPUT_DIMENSION))

train_domains_names = ['t1', 't2', 't3']
val_domains_names = ['v1', 'v2', 'v3']

domains = {}
domain_lengths = {}
for i in range(len(train_domains_names)):
    l1, l2 = np.random.randint(4, 7), np.random.randint(4, 7)
    domains[train_domains_names[i]] = simulate_domain(l1)
    domains[val_domains_names[i]] = simulate_domain(l2)
    
    domain_lengths[train_domains_names[i]], domain_lengths[val_domains_names[i]] = l1, l2

def andyho_magia(domain, indices):
    prepare_input = torch.empty(CROP_SIZE**2, 1, INPUT_LAYERS, INPUT_DIMENSION, INPUT_DIMENSION)
    for i, ijpair in enumerate(indices):
        prepare_input[i] = domains[domain][ijpair[0], ijpair[1]]#.view(1, INPUT_LAYERS, INPUT_DIMENSION, INPUT_DIMENSION)
        
    outputs = torch.randint(63, (CROP_SIZE**2,))
    return prepare_input, outputs
 
#%%
af = AlphaFold()
opt = torch.optim.Adam(af.parameters()) 
af.fit(train_domains_names, val_domains_names, opt)

    