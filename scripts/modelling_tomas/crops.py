#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating Crops. !!!Needs to be Redone to account for using torch
tensors instead of numpy arrays!!!
"""

import numpy as np
import torch

def calc_borders(L, offset=[0, 0], c=64):
    """Calculates and returns the indices of crop borders given:
    L      : Length of protein
    offset : list of two integers telling how to offset in [x, y] axis
    c      : crop size
    """
    # What is the maximum offset from the top left corner, in order to
    # stay in the padded area, and cover the entire protein?
    k = L // c + 1

    middle = L // 2 - (c // 2) * (k % 2)

    before_middle = middle // c
    after_middle = middle // c + (k % 2)

    borders = []

    for i in range(before_middle):
        borders.append(middle - c * (i + 1))

    borders.append(middle)

    for i in range(after_middle):
        borders.append(middle + c * (i + 1))

    border_indices = np.empty((len(borders) ** 2, 2), dtype=np.int)

    for i in range(len(borders)):
        for j in range(len(borders)):
            border_indices[len(borders) * i + j] = [borders[i], borders[j]]

    return border_indices + offset


def offset_borders(L, c=64, random_state=1):
    """Calculates allowed offsets for a given 'L' and 'c' and returns the offsetted borders"""

    k = L // c + 1
    m = k * c - c // 2

    if L % c == 0:
        if random_state % 2 == 0:
            return calc_borders(L, c=c), k
        else:
            k = L // c
            borders0 = np.arange(c, L, c)
            borders = []
            for i in range(k - 1):
                for j in range(k - 1):
                    borders.append([borders0[i], borders0[j]])
            return np.array(borders), k

    np.random.seed(random_state)

    # we have to make sure that we stay within the padded area (+- c / 2)
    max_allowed_offset = c // 2 - np.abs(m - L)

    left_up, right_down = max_allowed_offset // 2, max_allowed_offset // 2 + max_allowed_offset % 2

    allowed_offsets = np.arange(-left_up, right_down + 1)

    random_offset = np.random.choice(allowed_offsets, (1, 2))

    borders0 = calc_borders(L, random_offset, c)
    borders = np.concatenate([
        np.zeros((k ** 2 - (k - 1) ** 2, 2)),
        borders0])

    return calc_borders(L, random_offset, c), k


def make_crop_indices(L, c=64, random_state=1):
    """Function returns indices for designing crops:
    Returns: [
        [(i0, j0), (i, j), 'topleft'], # last entry tells about the padding
        [...]
    ]

    should be of length k**2 == number of crops
    """

    if L < c:
        return []

    if L == c and random_state % 2 == 1:
        return np.array([[(0, 0), (c, c), None]])

    offsets, k = offset_borders(L, c=c, random_state=random_state)

    # create a 3D array - (k+1, k+1, 2) with border indices
    i_unique = np.concatenate((np.unique(offsets[:, 0]), [L]))
    j_unique = np.concatenate((np.unique(offsets[:, 1]), [L]))

    offsets2 = np.zeros((k + 1, k + 1, 2))

    for i in range(1, k + 1):
        offsets2[i, :, 0] = np.repeat(i_unique[i - 1], k + 1)
        offsets2[:, i, 1] = np.repeat(j_unique[i - 1], k + 1)

    crop_indices = np.empty((k**2, 3), dtype='O')

    for m in range(k):
        for n in range(k):

            (i0, j0), (i, j) = offsets2[m, n, :].astype(np.int), offsets2[m + 1, n + 1, :].astype(np.int)

            if i - i0 == c and j - j0 == c:
                padding = None
            else:
                # TOP
                if i0 == 0:
                    # Top-left corner
                    if j0 == 0:
                        if i - i0 == c:
                            padding = 'left'
                        elif j - j0 == c:
                            padding = 'top'
                        else:
                            padding = 'topleft'

                    # between top corners
                    elif j < L and i < c:
                        padding = 'top'

                    # Top-right corner
                    elif j == L:
                        if i - i0 == c:
                            padding = 'right'
                        elif j - j0 == c:
                            padding = 'top'
                        else:
                            padding = 'topright'
                # BOTTOM
                elif i == L:
                    # Bottom-left corner
                    if j0 == 0:
                        if i - i0 == c:
                            padding = 'left'
                        elif j - j0 == c:
                            padding = 'bottom'
                        else:
                            padding = 'bottomleft'

                    # between Bottom corners
                    elif j < L and (i - i0) < c:
                        padding = 'bottom'

                    # Bottom-right corner
                    elif j == L:
                        if i - i0 == c:
                            padding = 'right'
                        elif j - j0 == c:
                            padding = 'bottom'
                        else:
                            padding = 'bottomright'

                elif j0 == 0:
                    if i < L:
                        padding = 'left'
                else:
                    padding = 'right'

            crop_indices[m * k + n] = [(i0, j0), (i, j), padding]

    return crop_indices


def pad_crop(crop, edges, output_crop=None, crop_size=64, random_state=1):
    """Creates 0 padding along selected edge(s)"""

    if edges is None:
        return crop

    else:
        np.random.seed(random_state)
        padded_input = np.zeros((crop.shape[0], crop_size, crop_size))

        # Case when L < c: place the crop inside zero matrix (c, c)
        if edges == 'all':
            padded_output = np.zeros((1, crop_size, crop_size))
            
            L = crop.shape[1]
            offset_range = np.arange(crop_size - L + 1)
            i_offset, j_offset = np.random.choice(offset_range), np.random.choice(offset_range)
            
            padded_input[:, i_offset:(L + i_offset), j_offset:(L + j_offset)] = crop
            padded_output[:, i_offset:(L + i_offset), j_offset:(L + j_offset)] = output_crop
            
            return padded_input, padded_output

        elif edges == 'topleft' or edges == 'top':
            padded_input[:, -crop.shape[1]:, -crop.shape[2]:] = crop
        elif edges == 'topright' or edges == 'right':
            padded_input[:, -crop.shape[1]:, :crop.shape[2]] = crop
        elif edges == 'bottomleft' or edges == 'left':
            padded_input[:, :crop.shape[1], -crop.shape[2]:] = crop
        elif edges == 'bottomright' or edges == 'bottom':
            padded_input[:, :crop.shape[1], :crop.shape[2]] = crop

    return padded_input


def make_batches(input_tensor, output_tensor, c=64, random_state=1):
    """Function should return input and output of shapes:
    input:  (crops, 675, 64, 64)
    output: (crops, 64, 64)
    """
    
    Ch, L = input_tensor.shape[0], input_tensor.shape[1]
    output_tensor = output_tensor.reshape((1, L, L))
    
    if L < c:
        input_batches, output_batches = pad_crop(input_tensor, 'all', output_tensor)
        return torch.from_numpy(input_batches).view((1, Ch, c, c)).to(torch.float32), torch.from_numpy(output_batches).to(torch.long)
    
    crop_indices = make_crop_indices(L, c=c, random_state=random_state)
    
    input_batches = np.empty((len(crop_indices), Ch, c, c))
    output_batches = np.empty((len(crop_indices), c, c))

    for m in range(len(crop_indices)):
        (i0, j0), (i, j), padding = crop_indices[m]
        input_batches[m, :, :, :] = pad_crop(input_tensor[:, i0:i, j0:j], padding, c)
        output_batches[m, :, :] = pad_crop(output_tensor[:, i0:i, j0:j], padding, c)

    return torch.from_numpy(input_batches).to(torch.float32), torch.from_numpy(output_batches).to(torch.long)
