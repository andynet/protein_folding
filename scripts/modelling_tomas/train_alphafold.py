from training_function import train
from AlphaFold import AlphaFold
import torch
import numpy as np

EPOCHS = 50
BATCH_SIZE = 8
ITERATION_DOMAINS = 500
VALIDATION_SIZE = 250
CPU_WORKERS = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0

train_domains = np.loadtxt('../../data/our_input/train_domains.csv', dtype='O')

model = AlphaFold().to("cuda")
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
#scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=800000, gamma=0.3)

train(model, opt, '../../steps/alphafold_results', train_domains, 
      EPOCHS=EPOCHS,
      BATCH_SIZE=BATCH_SIZE,
      ITERATION_DOMAINS=ITERATION_DOMAINS,
      CPU_WORKERS=CPU_WORKERS
     )

