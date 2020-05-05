#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Training function with AlphaFolds Cropping scheme
"""

import numpy as np
import pickle
import torch
from helper_functions import *
import time

EPOCHS = 10
BATCH_SIZE = 8

ITERATION_DOMAINS = 500
VALIDATION_SIZE = 250

CPU_WORKERS = 16

# %%
def train(model, optimizer,
          datapath,
          domains,
          scheduler=None,
          EPOCHS=EPOCHS,
          BATCH_SIZE=BATCH_SIZE,
          ITERATION_DOMAINS=ITERATION_DOMAINS,
          VALIDATION_SIZE=VALIDATION_SIZE,
          CPU_WORKERS=CPU_WORKERS,
          device="cuda"):
    
   
    history = []
    min_loss = 100

    with open('../../steps/domain_lengths.pkl', 'rb') as f:
        lengths = pickle.load(f)   
    
    for epoch in range(EPOCHS):
        start = time.time()
        np.random.seed(1618 + epoch)
        domain_list = np.random.permutation(domains)
        
        validation_domains = np.random.choice(domain_list, VALIDATION_SIZE, replace=False)
        train_domains = np.setdiff1d(domain_list, validation_domains)
        train_domains_eval = np.random.choice(train_domains, VALIDATION_SIZE, replace=False)
        
        for iteration in range(len(train_domains) // ITERATION_DOMAINS):
            s = time.time()
            iteration_domains = train_domains[(iteration * ITERATION_DOMAINS):((iteration + 1) * ITERATION_DOMAINS)]
            
            X, Y = load_crops(iteration_domains, lengths, random_state=1618+epoch, num_workers=CPU_WORKERS)
            
            # FIT
            for b in range(X.shape[0] // BATCH_SIZE):
                # create batch
                batch_ind = np.random.choice(np.arange(X.shape[0]), BATCH_SIZE)
                batch_in, batch_out = X[batch_ind, :, :, :], Y[batch_ind, :, :]

                # put it on GPU if available
                if device == 'cuda':
                    batch_in, batch_out = batch_in.to("cuda"), batch_out.to("cuda")
                    
                    model.fit(batch_in, batch_out, optimizer, scheduler)
                    
                    del batch_in, batch_out
                    torch.cuda.empty_cache()
                    
                else:
                    model.fit(batch_in, batch_out, optimizer)
            print(f'Iteration {iteration} done. Time: {time.time() - s}')        
        # EVALUATE
        # Train
        train_loss = evaluate(model, train_domains_eval, lengths, 
                              random_state=1618 + epoch, num_workers=CPU_WORKERS, device=device)
        val_loss = evaluate(model, validation_domains, lengths, 
                              random_state=1618 + epoch, num_workers=CPU_WORKERS, device=device)
        history.append([epoch, 
                        train_loss, 
                        val_loss
                        ])
        end = time.time()
        print('Epoch {:2d}, Time: {:.2f}, Train Loss: {:.4f}, Validation Loss: {:.4f}'
                .format(epoch, end - start, train_loss, val_loss))
        
        np.savetxt(f'{datapath}/history.csv', history, delimiter=',', fmt='%.3f')
        
        if val_loss < min_loss:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1
        #                # 'archs':args.arch
                        },
                       f'{datapath}/model.pth')
    #return np.array(history)
