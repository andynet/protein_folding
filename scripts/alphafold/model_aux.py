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
class AlphaFold(torch.nn.Module):

    def __init__(self, input_size, up_size, down_size, output_size,
                 aux_size, crop_size, RESNET_depth):
        super(AlphaFold, self).__init__()
        self.params = {
            'input_size': input_size,
            'up_size': up_size,
            'down_size': down_size,
            'output_size': output_size,
            'aux_size': aux_size,
            'crop_size': crop_size,
            'RESNET_depth': RESNET_depth
        }

        self.num_updates = 0
        # self.crop_size = crop_size
        # self.output_size = output_size

        self.bn1 = torch.nn.BatchNorm2d(num_features=input_size)

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_size, out_channels=up_size, kernel_size=1)

        self.RESNETBlocks = torch.nn.ModuleList(
            [RESNETBlock(up_size=up_size, down_size=down_size,
                         dilation_size=2**(i % 4)) for i in range(0, RESNET_depth)])

        self.conv2_1 = torch.nn.Conv2d(
            in_channels=up_size, out_channels=output_size, kernel_size=1)

        self.conv2_2 = torch.nn.Conv2d(
            in_channels=up_size, out_channels=aux_size, kernel_size=1)

        self.conv3 = torch.nn.Conv2d(
            in_channels=aux_size, out_channels=aux_size,
            kernel_size=(crop_size, 1),
            groups=aux_size)
        # self.conv3 = torch.nn.Conv2d(in_channels=82, out_channels=82, kernel_size=(162, 1), groups=82)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        for block in self.RESNETBlocks:
            x = block(x)

        dist = self.conv2_1(x)
        aux = self.conv3(self.conv2_2(x))
        aux = torch.squeeze(aux, dim=2)
        ss = aux[:, 0:8, :]
        phi = aux[:, 8:45, :]
        psi = aux[:, 45:82, :]
        x = (dist, ss, phi, psi)
        return x

    def fit(self, X, Y, criterion, optimizer):
        """
        X.shape = (B, C, H, W)
        X.dtype = torch.float32

        Y = (dist, ss, phi, psi)
        dist.shape = (B, H, W)
        ss.shape = (B, H)
        phi.shape = (B, H)
        psi.shape = (B, H)
        .dtype = torch.int64
        """

        self.train()
        optimizer.zero_grad()
        dist_pred, ss_pred, phi_pred, psi_pred = self.forward(X)
        dist, ss, phi, psi = Y

        loss = 10 * criterion(dist_pred, dist)
        loss += 2 * criterion(ss_pred, ss)
        loss += 2 * criterion(phi_pred, phi)
        loss += 2 * criterion(psi_pred, psi)
        loss.backward()
        optimizer.step()
        self.num_updates += 1

    def predict(self, X):
        """
        X.shape = (B, C, H, W)
        """
        self.eval()
        B, C, H, W = X.shape
        output_size = self.params['output_size']
        crop_size = self.params['crop_size']
        aux_size = self.params['aux_size']

        dist_predicted = torch.zeros((B, output_size, H, W))
        dist_count = torch.zeros((B, output_size, H, W))
        ones = torch.ones((B, output_size, crop_size, crop_size))

        aux_predicted = torch.zeros((B, aux_size, H))
        aux_count = torch.zeros((B, aux_size, H))
        ones2 = torch.ones((B, aux_size, crop_size))

        for i in range(0, H - crop_size + 1):
            for j in range(0, W - crop_size + 1):
                i_slice = slice(i, i + crop_size)
                j_slice = slice(j, j + crop_size)

                prediction = self.forward(X[:, :, i_slice, j_slice])
                dist, ss, phi, psi = prediction

                dist_predicted[:, :, i_slice, j_slice] += dist.cpu()
                dist_count[:, :, i_slice, j_slice] += ones

                aux_predicted[:, 0:8, i_slice] += ss.cpu()
                aux_predicted[:, 8:45, i_slice] += phi.cpu()
                aux_predicted[:, 45:82, i_slice] += psi.cpu()
                aux_count[:, :, i_slice] += ones2

        dist_predicted /= dist_count
        aux_predicted /= aux_count
        return dist_predicted, aux_predicted

    def score(self, X, Y, criterion):
        """
        X.shape = (B, C, H, W)
        Y.shape = (B, H, W)
        """
        self.eval()
        dist_pred, ss_pred, phi_pred, psi_pred = self.forward(X)
        dist, ss, phi, psi = Y

        loss = 10 * criterion(dist_pred, dist)
        loss += 2 * criterion(ss_pred, ss)
        loss += 2 * criterion(phi_pred, phi)
        loss += 2 * criterion(psi_pred, psi)
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
