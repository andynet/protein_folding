#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:14:26 2020

@author: andyb
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
            kernel_size=3, padding=dilation_size,
            dilation=dilation_size, groups=down_size)

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
class AuxBlock(torch.nn.Module):
    def __init__(
        self, C_in, C_aux, crop_size
    ):
        super(AuxBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=C_in, out_channels=C_aux, kernel_size=1)

        self.conv2 = torch.nn.Conv2d(
            in_channels=C_aux,
            out_channels=C_aux,
            kernel_size=(crop_size, 1),
            groups=C_aux
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.squeeze(x, dim=2)
        return x


# %%
class AlphaFold(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        dist_channels, ss_channels, phi_channels, psi_channels,
        up_channels, down_channels,
        RESNET_depth,
        crop_size,
        weights
    ):
        super(AlphaFold, self).__init__()
        self.num_updates = 0
        self.weights = weights

        self.bn1 = torch.nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=up_channels, kernel_size=1)

        self.RESNETBlocks = torch.nn.ModuleList(
            [RESNETBlock(up_size=up_channels, down_size=down_channels,
                         dilation_size=2**(i % 4)) for i in range(0, RESNET_depth)])

        self.dist_aux = torch.nn.Conv2d(
            in_channels=up_channels, out_channels=dist_channels, kernel_size=1)

        self.ss_aux = AuxBlock(
            C_in=up_channels, C_aux=ss_channels, crop_size=crop_size)

        self.phi_aux = AuxBlock(
            C_in=up_channels, C_aux=phi_channels, crop_size=crop_size)

        self.psi_aux = AuxBlock(
            C_in=up_channels, C_aux=psi_channels, crop_size=crop_size)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        for block in self.RESNETBlocks:
            x = block(x)

        dist = self.dist_aux(x)
        ss = self.ss_aux(x)
        phi = self.phi_aux(x)
        psi = self.psi_aux(x)

        return dist, ss, phi, psi

    def fit(self, X, Y, criterion, optimizer):
        optimizer.zero_grad()
        loss = self.score(X, Y, criterion)
        loss.backward()
        optimizer.step()
        self.num_updates += 1

    def predict(self, X):
        return self.forward(X)

    def score(self, X, Y, criterion):
        Y_pred = self.predict(X)

        loss = (
            self.weights['dgram'] * criterion(Y_pred[0], Y[0])
            + self.weights['2nd'] * criterion(Y_pred[1], Y[1])
            + self.weights['phi'] * criterion(Y_pred[2], Y[2])
            + self.weights['psi'] * criterion(Y_pred[3], Y[3])
        )
        return loss


# %%
class Domains(torch.utils.data.Dataset):

    def __init__(self, domains, X_files, Y_files, verbosity=1):

        self.domains = domains
        self.X_files = X_files
        self.Y_files = Y_files
        self.verbosity = verbosity

    def __getitem__(self, idx):

        X = torch.load(self.X_files.format(domain=self.domains[idx]))
        X = X.to(dtype=torch.float32)

        Y = torch.load(self.Y_files.format(domain=self.domains[idx]))
        dist, ss, phi, psi = Y
        ss = ss.to(dtype=torch.int64) - 1
        Y = (dist, ss, phi, psi)

        if self.verbosity == 1:
            print(f'Domain {self.domains[idx]} loaded.')

        return X, Y, self.domains[idx]

    def __len__(self):
        return len(self.domains)
