#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Template
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 569
CHANNELS = 128
OUTPUT_BINS = 32
HIDDEN = 16
KERNEL_SIZE = 5
PADDING = 2
DILATION = 1
#GRADIENT_CLIP_MAX_NORM = 1

class ConvNet_aux(nn.Module):

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
        
        self.conv_dist = nn.Conv2d(CHANNELS, OUTPUT_BINS, 1)
        
        self.conv_aux0 = nn.Conv2d(CHANNELS, 83, 1)
        self.conv_aux_ij = nn.Conv2d(83, 83, (64, 1), groups=83)
        
    def forward(self, x):
        x = self.bnInp(x)
        x = torch.relu(self.bn1(self.convInp(x)))
        
        for i in range(self.HIDDEN):
            x = torch.relu(self.bnlist[i](self.convlist[i](x)))
        
        # distogram branch
        dist_out = F.log_softmax(self.conv_dist(x), dim=1)
        
        # auxiliary outputs branch
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
