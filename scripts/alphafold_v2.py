#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:57:00 2020

@author: andy
Based on: ProSPr: Democratized Implementation of Alphafold Protein Distance
Prediction Network - https://doi.org/10.1101/830273
"""

# %%
import torch


# %%
class RESNETBlock(torch.nn.Module):

    def __init__(self, up_size, down_size, dilation_size):
        super(RESNETBlock, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(num_features=up_size)

        self.project_down = torch.nn.Conv2d(
            in_channels=up_size, out_channels=down_size, kernel_size=1)

        self.bn2 = torch.nn.BatchNorm2d(num_features=down_size)

        self.dilation = torch.nn.Conv2d(
            in_channels=down_size, out_channels=down_size,
            kernel_size=3, dilation=dilation_size, padding=dilation_size)

        self.bn3 = torch.nn.BatchNorm2d(num_features=down_size)

        self.project_up = torch.nn.Conv2d(
            in_channels=down_size, out_channels=up_size, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.bn1(x)
        x = torch.nn.functional.elu(x)
        x = self.project_down(x)
        x = self.bn2(x)
        x = torch.nn.functional.elu(x)
        x = self.dilation(x)
        x = self.bn3(x)
        x = self.project_up(x)
        x = x + identity
        return x


# %%
class AlphaFold(torch.nn.Module):

    def __init__(self, input_size, up_size, down_size, output_size, RESNET_depth):
        super(AlphaFold, self).__init__()

        self.num_updates = 0

        self.bn1 = torch.nn.BatchNorm2d(num_features=input_size)

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_size, out_channels=up_size, kernel_size=1)

        self.RESNETBlocks = torch.nn.ModuleList(
            [RESNETBlock(up_size=up_size, down_size=down_size,
                         dilation_size=2**(i % 4)) for i in range(0, RESNET_depth)])

        self.conv2 = torch.nn.Conv2d(
            in_channels=up_size, out_channels=output_size, kernel_size=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        for block in self.RESNETBlocks:
            x = block(x)
        x = self.conv2(x)
        return x
