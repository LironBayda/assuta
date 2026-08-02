
import numpy as np
import nibabel as nib
from os import listdir
from os.path import join
from typing import Tuple, List


def list_image_paths(mypath: str, prefix: str) -> list:
    """Return sorted list of image paths containing the prefix."""
    return sorted([join(mypath, f) for f in listdir(mypath) if prefix in f])


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return its data array."""
    return nib.load(path).get_fdata()


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Handle NaNs/Infs and reshape 5D images to 4D with time-first order."""
    images = np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
    if images.ndim == 5:  # If 5D, collapse to 4D (time, x, y, z)
        images = np.transpose(images[0], (3, 0, 1, 2))
    return images


def get_images(mypath: str, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess images from a folder with a given prefix."""
    paths = list_image_paths(mypath, prefix)
    if not paths:
        raise FileNotFoundError(f"No images found with prefix '{prefix}' in {mypath}")

    first_image = load_nifti(paths[0])

    # If multiple images, stack them; else just use the first
    images = np.stack([load_nifti(p) for p in paths]) if len(paths) > 1 else first_image

    images = preprocess_images(images)
    affine = nib.load(paths[0]).affine
    return images, affine


def load_image_series(
    paths: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a series of NIfTI images from a directory based on filename prefix patterns.

    This function searches for files in `directory` that contain BOTH strings in `prefix`.
    It loads the selected subset of images, converts them to floating-point arrays,
    replaces non-finite values (NaN, Inf), and returns the full dynamic array and
    affine transformation matrix.

    Parameters
    ----------
    paths : List of strings
        Paths to the  NIfTI files.

    Returns
    -------
    images : np.ndarray
        4D array of shape [N, X, Y, Z]
    affine : np.ndarray
        4x4 affine matrix from the first loaded image.

    Raises
    ------
    ValueError
        If directory is invalid, prefixes are missing, or no files match filter.
    IndexError
        If slice indices are out of range.
    """

    # ----------------------------------------------------------------------
    if not isinstance(paths, list):
        raise ValueError("paths must be a list.")

    # ----------------------------------------------------------------------
    # Load all images
    # ----------------------------------------------------------------------
    loaded_images = []
    for file_path in paths:
        img_obj = nib.load(file_path)
        loaded_images.append(img_obj.get_fdata(dtype=np.float64))
    images = np.asarray(loaded_images, dtype=np.float64)

    # ----------------------------------------------------------------------
    # Replace non-finite values (NaN + Inf)
    # ----------------------------------------------------------------------
    non_finite_mask = ~np.isfinite(images)
    if np.any(non_finite_mask):
        images[non_finite_mask] = 0.0

    # Affine from the first file
    first_affine = nib.load(paths[0]).affine

    return images[0], first_affine

