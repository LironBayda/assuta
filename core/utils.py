import os
from glob import glob
from typing import List

import numpy as np
import torch
from torch import nn

from io_utils.nifti_utils import load_image_series
import nibabel as nib

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

def center_crop_nifti(data, affine, target_shape=(64, 64, 20)):
    """
    Center-crop a NIfTI image and correctly update its affine.

    Parameters
    ----------
    data : ndarray
        Input image data.
    affine : ndarray
        Input image affine.
    target_shape : tuple
        Desired shape (X, Y, Z).

    Returns
    -------
    (ndarray, ndarray)
        Cropped data and corrected affine.
    """

    original_shape = np.array(data.shape[:3])
    target_shape = np.array(target_shape)

    if np.any(original_shape < target_shape):
        raise ValueError(
            f"Image shape {tuple(original_shape)} is smaller than "
            f"target shape {tuple(target_shape)}"
        )

    # Calculate crop starting coordinates
    start = (original_shape - target_shape) // 2
    end = start + target_shape

    # Crop
    cropped = data[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ]

    # Update translation.
    # The new voxel (0,0,0) corresponds to the old voxel `start`.
    new_origin = nib.affines.apply_affine(affine, start)

    new_affine = affine.copy()
    new_affine[:3, 3] = new_origin

    return cropped, new_affine


def find_files(base_path, pattern):
    """
    Return a list of matching files using join + glob.
    """
    return glob(os.path.join(base_path, pattern))




