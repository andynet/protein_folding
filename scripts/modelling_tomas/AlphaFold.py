#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Template
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 569

CHANNELS1 = 256
NUM_BLOCKS1 = 4

CHANNELS2 = 128
NUM_BLOCKS2 = 32

OUTPUT_BINS = 32
#GRADIENT_CLIP_MAX_NORM = 1


class AlphaUnit(nn.Module):
    """ Dilation Block
    Creates a block described by the image in the paper with specific dilation
    """

    def __init__(self, in_channels, out_channels=None, dilation=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        else:
            # Changing number of channels from 256 -> 128
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.downbn = nn.BatchNorm2d(out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.bn1 = nn.BatchNorm2d(in_channels, out_channels)
        self.projectdown = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.conv = nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=dilation, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(in_channels // 2)
        self.projectup = nn.Conv2d(in_channels // 2, out_channels, 1)

    def forward(self, x):
        if self.out_channels == self.in_channels:
            x0 = x
        else:
            x0 = self.downbn(self.downsample(x))

        x = F.elu(self.bn1(x))
        x = self.projectdown(x)
        x = F.elu(self.bn2(x))
        x = self.conv(x)
        x = F.elu(self.bn3(x))
        x = self.projectup(x)
        return x + x0


class AlphaBlock(nn.Module):
    """Generates Set of dilation blocks with different dilation values"""

    def __init__(self, in_channels):
        super().__init__()
        alphaunit = AlphaUnit

        self.alphaunit1 = alphaunit(in_channels, dilation=1)
        self.alphaunit2 = alphaunit(in_channels, dilation=2)
        self.alphaunit4 = alphaunit(in_channels, dilation=4)
        self.alphaunit8 = alphaunit(in_channels, dilation=8)

    def forward(self, x):
        x = self.alphaunit1(x)
        x = self.alphaunit2(x)
        x = self.alphaunit4(x)
        x = self.alphaunit8(x)
        return x


class AlphaFold(nn.Module):

    def __init__(self,
                 CHANNELS1=CHANNELS1,
                 NUM_BLOCKS1=NUM_BLOCKS1,
                 CHANNELS2=CHANNELS2,
                 NUM_BLOCKS2=NUM_BLOCKS2,
                 OUTPUT_BINS=OUTPUT_BINS):
        
        super().__init__()
        
        alphablock = AlphaBlock
        alphaunit = AlphaUnit
        
        self.bnInp = nn.BatchNorm2d(INPUT_CHANNELS)
        self.convInp = nn.Conv2d(INPUT_CHANNELS, CHANNELS1, 1)
        self.bn1 = nn.BatchNorm2d(CHANNELS1)
        
        # Block 1 Module list
        self.afblock_list1 = nn.ModuleList([alphablock(CHANNELS1) for i in range(NUM_BLOCKS1)])
        
        # Changing number of channels from 256 -> 128
        self.downsample = alphaunit(CHANNELS1, CHANNELS2)
        
        # Block 2 Module list
        self.afblock_list2 = nn.ModuleList([alphablock(CHANNELS2) for i in range(NUM_BLOCKS2)])
        
        self.conv_dist = nn.Conv2d(CHANNELS2, OUTPUT_BINS, 1)
        
        self.conv_aux0 = nn.Conv2d(CHANNELS2, 83, 1)
        self.conv_aux_ij = nn.Conv2d(83, 83, (64, 1), groups=83)
        
    def forward(self, x):
        x = self.bnInp(x)
        x = F.elu(self.bn1(self.convInp(x)))
        
        # Cycling through blocks with channels = CHANNELS1
        for block in self.afblock_list1:
            x = block(x)
            
        # Downsampling
        x = self.downsample(x)
        
        # Cycling through blocks with channels = CHANNELS2
        for block in self.afblock_list2:
            x = block(x)
        
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
