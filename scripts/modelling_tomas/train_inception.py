from training_function import train
from Inception_aux import Inception_aux
import torch
import numpy as np

EPOCHS = 20
BATCH_SIZE = 8
ITERATION_DOMAINS = 500
VALIDATION_SIZE = 250
CPU_WORKERS = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0

train_domains = np.loadtxt('../../data/our_input/train_domains.csv', dtype='O')

model = Inception_aux().to("cuda")
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100000, gamma=0.5)

train(model, opt, '../../steps/inception_results/scheduler', train_domains, scheduler=scheduler,
      EPOCHS=EPOCHS,
      BATCH_SIZE=BATCH_SIZE,
      ITERATION_DOMAINS=ITERATION_DOMAINS,
      CPU_WORKERS=CPU_WORKERS
     )
