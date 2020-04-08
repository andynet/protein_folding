#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Template
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        return F.log_softmax(self.convOut, dim=0)

    def fit(self, X, Y, optimizer):
        self.train()
        optimizer.zero_grad()

        preds = self.forward(X)
        loss = F.nll_loss(preds, Y)

        loss.backward()
        optimizer.step()

    def predict(self, X):
        self.eval()
        return self.forward(X)
