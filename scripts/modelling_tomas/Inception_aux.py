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
INCEPTIONS = 64
OUTPUT_BINS = 32
#GRADIENT_CLIP_MAX_NORM = 1


class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, groups=in_channels)
        self.conv3x3_2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels)
        self.conv3x3_4 = nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4, groups=in_channels)
        self.conv3x3_8 = nn.Conv2d(in_channels, in_channels, 3, padding=8, dilation=8, groups=in_channels)

    def forward(self, x):
        i = self.conv1x1(x)
        c1 = self.conv3x3_1(i)
        c2 = self.conv3x3_2(i)
        c4 = self.conv3x3_4(i)
        c8 = self.conv3x3_8(i)
        o = c1 + c2 + c4 + c8 + x
        return o


class Inception_aux(nn.Module):

    def __init__(self,
                INPUT_CHANNELS=INPUT_CHANNELS,
                CHANNELS=CHANNELS,
                INCEPTIONS=INCEPTIONS,
                OUTPUT_BINS=OUTPUT_BINS):
        
        super().__init__()
        inception = InceptionModule
        
        self.inceptions = INCEPTIONS
        
        self.bnInp = nn.BatchNorm2d(INPUT_CHANNELS)
        self.convInp = nn.Conv2d(INPUT_CHANNELS, CHANNELS, 1)

        self.bn1 = nn.BatchNorm2d(CHANNELS)
        self.inception_list = nn.ModuleList(
            [inception(CHANNELS) for i in range(INCEPTIONS)]
        )
        self.bn_list = nn.ModuleList(
            [nn.BatchNorm2d(CHANNELS) for i in range(INCEPTIONS)]
        )

        self.conv_dist = nn.Conv2d(CHANNELS, OUTPUT_BINS, 1)
        
        self.conv_aux0 = nn.Conv2d(CHANNELS, 83, 1)
        self.conv_aux_ij = nn.Conv2d(83, 83, (64, 1), groups=83)


    def forward(self, x):
        x = self.bnInp(x)
        x = F.elu(self.bn1(self.convInp(x)))
        
        for i in range(self.inceptions):
            x = F.elu(self.bn_list[i](self.inception_list[i](x)))
        
        dist_out = F.log_softmax(self.conv_dist(x), dim=1)
        
        
        aux = self.conv_aux0(x)
        aux_j = self.conv_aux_ij(aux)
        aux_i = self.conv_aux_ij(torch.transpose(aux, 2, 3))
                        
        # distogram, secondary_(i, j), phi_(i, j), psi_(i, j)
        return dist_out,\
               F.log_softmax(aux_i[:, :9], dim=1), F.log_softmax(aux_j[:, :9], dim=1),\
               F.log_softmax(aux_i[:, 9:(9 + 37)], dim=1), F.log_softmax(aux_j[:, 9:(9 + 37)], dim=1),\
               F.log_softmax(aux_i[:, (9 + 37):(9 + 2 * 37)], dim=1), F.log_softmax(aux_j[:, (9 + 37):(9 + 2 * 37)], dim=1)

    def fit(self, X, Y, optimizer, scheduler):
        dmat, sec_i, sec_j, phi_i, phi_j, psi_i, psi_j = Y[:, :64, :],\
                                                         Y[:, 64:65, :], Y[:, 65:66, :],\
                                                         Y[:, 66:67, :], Y[:, 67:68, :],\
                                                         Y[:, 68:69, :], Y[:, 69:, :]
        self.train()
        optimizer.zero_grad()
        
        p_dmat, p_sec_i, p_sec_j, p_phi_i, p_phi_j, p_psi_i, p_psi_j = self.forward(X)
        loss = 10 * F.nll_loss(p_dmat, dmat)\
                  + F.nll_loss(p_sec_i, sec_i) + F.nll_loss(p_sec_j, sec_j)\
                  + F.nll_loss(p_phi_i, phi_i) + F.nll_loss(p_phi_j, phi_j)\
                  + F.nll_loss(p_psi_i, psi_j) + F.nll_loss(p_psi_i, psi_j)

        #torch.nn.utils.clip_grad_norm_(self.parameters(), GRADIENT_CLIP_MAX_NORM)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)
