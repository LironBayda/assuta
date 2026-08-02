"""
preprocessing.py

DCE-MRI SPGR preprocessing:
- T1 and M0 estimation from 2-point VFA
- Conversion of dynamic SPGR images to contrast concentration C(t)
"""

import numpy as np
from typing import Tuple

from config import SPGR, DCE
from core.create_blood_mask import make_blood_mask
from core.utils import find_files, get_tac_from_masked_region
from io_utils.jepg_utils import save_single_series_plot
from io_utils.nifti_utils import load_image_series
from registration import moving_to_reference_full



# ---------------------------
# T1 and M0 Estimation
# ---------------------------
def compute_T1_M0(
        s1: np.ndarray,
        s2: np.ndarray,
        alpha1_deg: float = SPGR["alpha_static_deg"][0],
        alpha2_deg: float = SPGR["alpha_static_deg"][1],
        TR: float = SPGR["TR_sec"],
        eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute T1 and M0 maps from two SPGR images with different flip angles.

    Parameters
    ----------
    s1, s2 : ndarray
        SPGR signals at flip angles alpha1 and alpha2.
    alpha1_deg, alpha2_deg : float
        Flip angles in degrees.
    TR : float
        Repetition time in seconds.
    eps : float
        Small value to prevent division by zero.

    Returns
    -------
    T1_0, M0 : ndarray
        Estimated T1 and M0 maps.
    """
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)

    if s1.shape != s2.shape:
        raise ValueError("s1 and s2 must have the same shape")

    a1, a2 = np.deg2rad(alpha1_deg), np.deg2rad(alpha2_deg)
    sin1, sin2 = np.sin(a1), np.sin(a2)
    cos1, cos2 = np.cos(a1), np.cos(a2)

    numerator = s1 / sin1 - s2 / sin2
    denominator = s1 / sin1 * cos1 - s2 / sin2 * cos2
    x = numerator / (denominator + eps)

    T1_0 = -TR / np.log(x + eps)
    M0 = s1 * (1 - x * cos1) / (sin1 * (1 - x) + eps)

    T1_0[np.isnan(T1_0)] = 0
    M0[np.isnan(M0)] = 0

    return T1_0, M0

# ---------------------------
# Dynamic SPGR -> Contrast Concentration
# ---------------------------
def dyn_spgr_to_concentration(
        dyn: np.ndarray,
        M0: np.ndarray,
        T1_0: np.ndarray,
        alpha_deg_dynamic: float = SPGR["alpha_dynamic_deg"],
        TR_dynamic: float = SPGR["TR_sec"],
        r1: float = SPGR["r1_relaxivity"]
) -> np.ndarray:
    """
    Convert dynamic SPGR series -> contrast agent concentration C(t) analytically.
    """
    dyn = np.asarray(dyn, float)
    M0 = np.asarray(M0, float)
    T1_0 = np.asarray(T1_0, float)

    alpha = np.deg2rad(alpha_deg_dynamic)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    # Reshape for broadcasting
    if dyn.ndim == 4:
        M0 = M0[..., None]
        T1_0 = T1_0[..., None]
    elif dyn.ndim == 2:
        M0 = M0[:, None]
        T1_0 = T1_0[:, None]
    else:
        raise ValueError("dyn must be 4D [X,Y,Z,T] or 2D [N_voxels,T]")

    with np.errstate(divide='ignore', invalid='ignore'):
        A = dyn / M0
        E1 = (sin_a - A) / (sin_a - A * cos_a)

    T1_dyn = np.full_like(E1, np.nan, float)
    valid = (E1 > 0) & (E1 < 1)
    T1_dyn[valid] = -TR_dynamic / np.log(E1[valid])

    with np.errstate(divide='ignore', invalid='ignore'):
        C_dyn = (1.0 / T1_dyn - 1.0 / T1_0) / r1

    C_dyn[~np.isfinite(C_dyn)] = 0
    return C_dyn


# ---------------------------
# High-Level Pipeline
# ---------------------------
def get_c_dyn(
        img2: np.ndarray,
        img15: np.ndarray,
        dyn_image: np.ndarray
) -> np.ndarray:
    """
    Compute dynamic contrast concentration from SPGR images.

    Parameters
    ----------
    img2, img15 : ndarray
        SPGR images at 2° and 15° for T1 estimation.
    dyn_image : ndarray
        Dynamic SPGR series [X,Y,Z,T] or [N_voxels,T].

    Returns
    -------
    c_dyn : ndarray
        Concentration time series.
    """
    T1_0, M0 = compute_T1_M0(img2, img15)
    c_dyn = dyn_spgr_to_concentration(dyn_image, M0, T1_0)
    return c_dyn


def preprocessing(path,
                  deg2_pattern="*_2*",
                  deg15_pattern="*_15*",
                  dyn_pattern="*dyns*"):
    """
    Full preprocessing pipeline:
    1. Load DCE + SPGR flip-angle images
    2. Register everything to DCE space
    3. Compute concentration-time curves
    4. Extract blood mask
    5. Extract AIF

    Returns
    -------
    c_dyn : ndarray (4D)
        Dynamic concentration image
    aif : ndarray (1D)
        Arterial input function
    affine : ndarray (4x4)
        Spatial affine (shared for all frames)
    """

    print(f"Processing {path}...")

    # -----------------------------
    # 1. Find input files
    # -----------------------------
    dyn_path   = find_files(path, dyn_pattern)[0]
    deg2_path  = find_files(path, deg2_pattern)[0]
    deg15_path = find_files(path, deg15_pattern)[0]

    # -----------------------------
    # 2. Register everything to DCE
    # -----------------------------
    deg2_img = moving_to_reference_full(
        deg2_path,
        dyn_path,
        run_affine=False,
        run_bspline=False
    )

    deg15_img = moving_to_reference_full(
        deg15_path,
        dyn_path,
        run_affine=False,
        run_bspline=False
    )

    dyn_img = moving_to_reference_full(
        dyn_path,
        dyn_path,
        run_affine=False,
        run_bspline=False
    )

    # -----------------------------
    # 3. Load affine correctly (DO NOT "convert" 4D → 3D)
    # -----------------------------
    _, affine = load_image_series([dyn_path])
    # ✅ Same affine is valid for all time frames

    # -----------------------------
    # 4. Compute dynamic concentration
    # -----------------------------
    c_dyn = get_c_dyn(
        deg2_img,
        deg15_img,
        dyn_img,
    )

    # -----------------------------
    # 5. Blood mask extraction
    # -----------------------------
    blood_mask = make_blood_mask(
        path,
        c_dyn,
        affine,
        DCE["blood_num_frames"]
    )

    # -----------------------------
    # 6. AIF extraction
    # -----------------------------
    aif = get_tac_from_masked_region(
        c_dyn,
        blood_mask
    )

    save_single_series_plot(
        path,
        np.cumsum(DCE["dt"]),
        aif,
        "DCE AIF",
        "Time (min)",
        "Concentration (mM)"
    )
    return c_dyn, aif, affine

