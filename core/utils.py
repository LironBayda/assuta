import os
from glob import glob
from typing import List

import numpy as np
import torch
from torch import nn

from io_utils.nifti_utils import load_image_series


def get_tac_from_masked_region(
        c_dyn: np.ndarray,
        mask: np.ndarray
) -> np.ndarray:


    mask = mask.astype(bool)

    valid = (mask > 0) & np.isfinite(c_dyn).all(axis=-1)

    # Extract: only positive concentration and inside mask
    voxel_curves = c_dyn[valid]  # shape: [N_voxels, T]

    # Keep only positive concentrations per voxel
    voxel_curves = np.where(voxel_curves > 0, voxel_curves, np.nan)

    # Mean over voxels → TAC(t)
    tac = np.nanmean(voxel_curves, axis=0)  # shape: [T]
    tac[np.isnan(tac)] = 0

    return tac


def init_network_xavier(net):
    """
    Apply Xavier initialization to all weights in the network.
    - GRU weight_ih and weight_hh → Xavier
    - Linear layers → Xavier
    - All biases → zeros
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)



def find_files(base_path, pattern):
    """
    Return a list of matching files using join + glob.
    """
    return glob(os.path.join(base_path, pattern))




