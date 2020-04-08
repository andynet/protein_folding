#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:36:31 2020

@author: tomasla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 569
CHANNELS = 128
INCEPTIONS = 16
OUTPUT_BINS = 32


class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        o = c1 + c2 + c3 + x
        return torch.relu(self.bn(o))


class Inception(nn.Module):

    def __init__(self,
                INPUT_CHANNELS=INPUT_CHANNELS,
                CHANNELS=CHANNELS,
                INCEPTIONS=INCEPTIONS,
                OUTPUT_BINS=OUTPUT_BINS):
        super().__init__()
        inception = InceptionModule

        self.bnInp = nn.BatchNorm2d(INPUT_CHANNELS)
        self.convInp = nn.Conv2d(INPUT_CHANNELS, CHANNELS, 1)

        self.bn1 = nn.BatchNorm2d(CHANNELS)
        self.inception_list = nn.ModuleList(
            [inception(CHANNELS) for i in range(INCEPTIONS)]
        )

        self.convOut = nn.Conv2d(CHANNELS, OUTPUT_BINS, 1)

    def forward(self, x):
        x = self.bnInp(x)
        x = torch.relu(self.bn1(self.convInp(x)))
        for inc in self.inception_list:
            x = inc(x)
            
        return F.log_softmax(self.convOut(x), dim=0)

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
