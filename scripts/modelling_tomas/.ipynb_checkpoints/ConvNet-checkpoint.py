#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:54:04 2020

@author: tomasla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 569
CHANNELS = 32
OUTPUT_BINS = 32
HIDDEN = 16
KERNEL_SIZE = 3
PADDING = 1
DILATION = 1

class ConvNet(nn.Module):

    def __init__(self,
                 INPUT_CHANNELS=INPUT_CHANNELS,
                 CHANNELS=CHANNELS,
                 HIDDEN=HIDDEN,
                 OUTPUT_BINS=OUTPUT_BINS,
                 KERNEL_SIZE=KERNEL_SIZE,
                 PADDING=PADDING,
                 DILATION=DILATION
                ):
        super().__init__()

        self.bnInp = nn.BatchNorm2d(INPUT_CHANNELS)
        self.convInp = nn.Conv2d(INPUT_CHANNELS, CHANNELS, 1)
        
        self.HIDDEN = HIDDEN
        self.bn1 = nn.BatchNorm2d(CHANNELS)

        self.convlist = nn.ModuleList([
            nn.Conv2d(CHANNELS, CHANNELS, KERNEL_SIZE, padding=PADDING, dilation=DILATION) for i in range(HIDDEN)
        ])

        self.bnlist = nn.ModuleList([
            nn.BatchNorm2d(CHANNELS) for i in range(HIDDEN)
        ])

        self.convOut = nn.Conv2d(CHANNELS, OUTPUT_BINS, 1)

    def forward(self, x):
        x = self.bnInp(x)
        x = torch.relu(self.bn1(self.convInp(x)))

        for i in range(self.HIDDEN):
            x = torch.relu(self.bnlist[i](self.convlist[i](x)))

        return F.log_softmax(self.convOut(x), dim=0)

    def fit(self, X, Y, optimizer):
        self.train()
        optimizer.zero_grad()

        preds = self.forward(X)
        loss = F.nll_loss(preds, Y)

        loss.backward()
        optimizer.step()
        del preds
        
    def predict(self, X):
        self.eval()
        return self.forward(X)
