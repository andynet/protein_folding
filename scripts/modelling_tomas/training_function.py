#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Training function with AlphaFolds Cropping scheme
"""
# %%
import numpy as np
from crops import make_batches
import os
import torch
import torch.nn.functional as F

# %%
EPOCHS = 10
BATCH_SIZE = 8

LOAD_DOMAINS = 500
VALIDATION_RATIO = 0.2


# %%
def train(model, optimizer, 
          #modelpath,
          EPOCHS=EPOCHS,
          BATCH_SIZE=BATCH_SIZE,
          LOAD_DOMAINS=LOAD_DOMAINS,
          VALIDATION_RATIO=VALIDATION_RATIO,
          device="cpu"):
    
    """Training Function
    
    Function for training neural networks. Function loads data in chunks of size
    "LOAD_DOMAINS" into memory. It picks training and validation domains 
    (VALIDATION_RATIO) and for both crops the data into 64x64 squares. These crops
    are then used to create batches of size BATCH_SIZE. Items inside of the batches
    are chosen randomly, meaning that some crops will be used multiple times.
    
    The input batch together with its output batch is then sent to a device (CPU or GPU)
    and the training is performed.
    """
    
    history = []
    min_loss = 100
    for epoch in range(EPOCHS):
        np.random.seed(1618 + epoch)
        domains = [f.split('_')[0] for f in os.listdir('../../data/our_input/tensors/')]

        # load "LOAD_DOMAINS" number of domains into the memory and train on them
        for it in range(len(domains) // LOAD_DOMAINS):

            tensors = {}
            outputs = {}
            iteration_domains = domains[(it * LOAD_DOMAINS):((it + 1) * LOAD_DOMAINS)]

            # Load tensors of corresponding domains to memory
            for domain in iteration_domains:
                tensors[domain] = torch.load(f'../../data/our_input/tensors/{domain}_X.pt')
                outputs[domain] = torch.load(f'../../data/our_input/distance_maps/distance_maps32/{domain}.pt')

            # create validation and train set
            validation_set_size = int(VALIDATION_RATIO * LOAD_DOMAINS)
            validation_domains = np.random.choice(iteration_domains, validation_set_size, replace=False)
            train_domains = np.setdiff1d(iteration_domains, validation_domains)

            # Generate the Tensors
            train_X = torch.tensor([], dtype=torch.float32)
            train_Y = torch.tensor([], dtype=torch.long)
            val_X = torch.tensor([], dtype=torch.float32)
            val_Y = torch.tensor([], dtype=torch.long)

            for domain in train_domains:
                i, o = make_batches(tensors[domain], outputs[domain])
                train_X = torch.cat((train_X, i))
                train_Y = torch.cat((train_Y, o))

            for domain in validation_domains:
                i, o = make_batches(tensors[domain], outputs[domain])
                val_X = torch.cat((val_X, i))
                val_Y = torch.cat((val_Y, o))

            print('Train crops: ', train_X.shape[0], 'Validation Crops: ', val_X.shape[0])

            # FIT
            for b in range(train_X.shape[0] // BATCH_SIZE):
                # create batch
                batch_ind = np.random.choice(np.arange(train_X.shape[0]), BATCH_SIZE)
                batch_in, batch_out = train_X[batch_ind, :, :, :], train_Y[batch_ind, :, :]

                # put it on GPU if available
                if device == 'cuda':
                    batch_in, batch_out = batch_in.to("cuda"), batch_out.to("cuda")
                    
                    model.fit(batch_in, batch_out, optimizer)
                    
                    del batch_in, batch_out
                    torch.cuda.empty_cache()
                    
                else:
                    model.fit(batch_in, batch_out, optimizer)

            # EVALUATE
            # Train
            train_losses = []
            train_preds = torch.tensor([])
            if device == 'cuda':
                
                for b in range(train_X.shape[0] // BATCH_SIZE):
                    batch_ind = np.arange(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
                    batch_in = train_X[batch_ind, :, :, :]
                    
                    batch_in = batch_in.to("cuda")
                    
                    with torch.no_grad():
                        batch_preds = model.predict(batch_in)
                    
                    train_preds = torch.cat((train_preds, batch_preds.to('cpu')))
                    
                    del batch_in, batch_preds
                    torch.cuda.empty_cache()
                
                train_loss = F.nll_loss(train_preds, 
                                        train_Y[torch.arange((train_X.shape[0] // BATCH_SIZE) * BATCH_SIZE),:,:]).item()
            else:
                train_preds = model.predict(train_X)
                train_loss = F.nll_loss(train_X, train_Y).item()
            
            # Validation
            val_losses = []
            val_preds = torch.tensor([])
            if device == 'cuda':
                for b in range(val_X.shape[0] // BATCH_SIZE):
                    batch_ind = np.arange(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
                    batch_in = val_X[batch_ind, :, :, :]
                    
                    batch_in = batch_in.to("cuda")
                    
                    with torch.no_grad():
                        batch_preds = model.predict(batch_in)
                    
                    val_preds = torch.cat((val_preds, batch_preds.to('cpu')))
                    
                    del batch_in, batch_preds
                    torch.cuda.empty_cache()
                
                val_loss = F.nll_loss(val_preds, 
                                      val_Y[torch.arange((val_X.shape[0] // BATCH_SIZE) * BATCH_SIZE),:,:]).item()
            else:
                val_preds = model.predict(val_X)
                val_loss = F.nll_loss(val_X, val_Y).item()
  
            history.append([epoch, 
                            min(it * LOAD_DOMAINS / len(domains), 100),
                            train_loss, 
                            val_loss
                           ])
            
            print('Epoch {:2d}, Domains Trained {:.2f}%, Train Loss: {:.4f}, Validation Loss: {:.4f}'
                  .format(epoch, min(it * LOAD_DOMAINS / len(domains), 100), train_loss, val_loss))
        #if val_loss < min_loss:
        #    torch.save({'model': model.state_dict(),
        #                'optimizer': optimizer.state_dict(),
        #                'epoch': epoch + 1
        #                # 'archs':args.arch
        #                }
        #               )
    return np.array(history)
    
