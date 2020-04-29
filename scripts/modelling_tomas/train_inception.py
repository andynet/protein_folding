from training_function import train
from ConvNet import ConvNet
from Inception import Inception
import torch
import numpy as np

EPOCHS = 20
BATCH_SIZE = 8
ITERATION_DOMAINS = 500
VALIDATION_SIZE = 250
CPU_WORKERS = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0

train_domains = np.loadtxt('../../data/our_input/train_domains.csv', dtype='O')

model = Inception().to("cuda")
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train(model, opt, '../../steps/inception_results', train_domains, 
      EPOCHS=EPOCHS,
      BATCH_SIZE=BATCH_SIZE,
      ITERATION_DOMAINS=ITERATION_DOMAINS,
      CPU_WORKERS=CPU_WORKERS
     )
