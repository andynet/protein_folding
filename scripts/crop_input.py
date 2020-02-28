#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:31:09 2020

@author: andyb
"""

# %%
import torch
import numpy as np


# %%
def get_batch(domain, batch_indices, crop_size):
    path = \
        f'/faststorage/project/deeply_thinking_potato/data/prospr/tensors/{domain}.pt'
    in_tensor, out_tensor = torch.load(path)

    len_seq = in_tensor.shape[0]
    crop_middle = crop_size // 2

    batch_in = torch.zeros([len(batch_indices), 1,
                            in_tensor.shape[2], crop_size, crop_size])

    batch_out = torch.zeros([len(batch_indices)])

    for i, (x, y) in enumerate(batch_indices):
        raw_bounds = [x - crop_middle, x + crop_middle,
                      y - crop_middle, y + crop_middle]

        clipped_bounds = [max(0, raw_bounds[0]), min(len_seq, raw_bounds[1]),
                          max(0, raw_bounds[2]), min(len_seq, raw_bounds[3])]

        image_bounds = [clipped_bounds[0] - x + crop_middle,
                        clipped_bounds[1] - x + crop_middle,
                        clipped_bounds[2] - y + crop_middle,
                        clipped_bounds[3] - y + crop_middle]

        tmp = in_tensor[clipped_bounds[0]:clipped_bounds[1],
                        clipped_bounds[2]:clipped_bounds[3],
                        :].permute(2, 0, 1)

        batch_in[i, 0, :,
                 image_bounds[0]:image_bounds[1],
                 image_bounds[2]:image_bounds[3]
                 ] = tmp

        batch_out[i] = out_tensor[x, y]

    return batch_in, batch_out.to(dtype=torch.int64)


# %%
if __name__ == "__main__":
    crop_size = 32
    domain = '1a6sA00'
    batch_indices = np.array([np.random.choice(np.arange(87), size=crop_size**2),
                              np.random.choice(np.arange(87), size=crop_size**2)]).T

    data = get_batch(domain, batch_indices, crop_size)
    # check with:
    # tmp = batch_in[0, 0, :, :, :].numpy()
