"""
Dictionary-Based Blood Signal Extraction for Dynamic Imaging

This module implements a dictionary-based method to identify blood voxels
from dynamic 4D imaging data (e.g., DCE-MRI or PET).

Algorithm Overview:
1. Build a fixed dictionary of representative TACs (time-activity curves):
   - Blood: Parker arterial input function (AIF)
   - Tissue: 1-compartment model response to AIF
   - Background: constant signal
2. Normalize dictionary atoms to unit norm.
3. Fit each voxel's TAC to the dictionary using non-negative least squares (NNLS)
   to obtain mixture weights (W_map).
4. Identify blood voxels based on high weight of the blood dictionary component
   and low weights for tissue/background components.
5. Restrict candidates to a plausible anatomical region and keep the largest
   connected component on each side of the midline, to enforce coherent
   vessel structures.
6. Optionally smooth the 4D image and/or erode the final mask to reduce noise.
7. Save the resulting blood mask as a NIfTI file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, gaussian_filter, label
from scipy.optimize import nnls

from config import BLOOD_DICTIONARY

logger = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed

    _HAVE_JOBLIB = True
except ImportError:  # joblib is an optional speed-up, not a hard dependency
    _HAVE_JOBLIB = False


# ---------------------------
# Parker AIF
# ---------------------------
def parker_aif(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Parker arterial input function (AIF).

    Parameters
    ----------
    t : ndarray
        Time vector, in minutes.

    Returns
    -------
    ndarray
        Blood plasma TAC evaluated at each time point in `t`.
    """
    A1, A2 = 0.809, 0.330
    T1, T2 = 0.170, 0.365
    sigma1, sigma2 = 0.0563, 0.132
    alpha, beta, s = 1.050, 0.1685, 38.078
    tau = 0.483

    gauss1 = A1 * np.exp(-((t - T1) ** 2) / (2 * sigma1 ** 2))
    gauss2 = A2 * np.exp(-((t - T2) ** 2) / (2 * sigma2 ** 2))
    sigmoid = alpha * np.exp(-beta * t) / (1 + np.exp(-s * (t - tau)))

    return gauss1 + gauss2 + sigmoid


# ---------------------------
# One-Tissue Compartment Model
# ---------------------------
def one_tcm_tissue(
    Cp: NDArray[np.float64], t: NDArray[np.float64], K1: float, k2: float
) -> NDArray[np.float64]:
    """
    Compute a tissue TAC from a 1-tissue-compartment model (1TCM).

    Parameters
    ----------
    Cp : ndarray
        Plasma input function, sampled on `t`.
    t : ndarray
        Time vector, in minutes, uniformly spaced.
    K1 : float
        Forward (blood-to-tissue) transfer rate constant.
    k2 : float
        Efflux (tissue-to-blood) rate constant.

    Returns
    -------
    ndarray
        Tissue TAC, same length as `t`.
    """
    dt = t[1] - t[0]
    kernel = np.exp(-k2 * t)
    Ct = K1 * np.convolve(Cp, kernel)[: len(t)] * dt
    return Ct


# ---------------------------
# Build Fixed Dictionary
# ---------------------------
def build_fixed_dictionary(
    T: int,
    dt_seconds: float | None = None,
    K1: float | None = None,
    k2: float | None = None,
    background_level: float = 200.0,
) -> NDArray[np.float64]:
    """
    Build a fixed dictionary of 3 template TACs (blood, tissue, background).

    Parameters
    ----------
    T : int
        Number of time frames.
    dt_seconds : float, optional
        Temporal resolution, in seconds. Defaults to
        ``BLOOD_DICTIONARY["dt_seconds"]`` at call time.
    K1 : float, optional
        Forward rate constant used for the tissue template. Defaults to
        ``BLOOD_DICTIONARY["K1"]`` at call time.
    k2 : float, optional
        Efflux rate constant used for the tissue template. Defaults to
        ``BLOOD_DICTIONARY["k2"]`` at call time.
    background_level : float
        Constant signal level for the background atom.

    Returns
    -------
    ndarray, shape (3, T)
        Unit-norm dictionary atoms: [blood, tissue, background].

    Notes
    -----
    The tissue atom is generated at a single fixed (K1, k2) pair. Real
    tissue kinetics vary voxel-to-voxel, so voxels whose true kinetics sit
    far from this template (in particular ones with unusually fast or slow
    washout) are more likely to be mismatched against the blood or
    background atoms instead. If that turns out to matter for your data,
    consider building several tissue atoms spanning a plausible (K1, k2)
    range rather than a single point estimate.

    `dt_seconds`, `K1`, and `k2` are resolved from `BLOOD_DICTIONARY` inside
    the function body (rather than as literal defaults) so that changes to
    `config.BLOOD_DICTIONARY` after import are picked up. Using
    `BLOOD_DICTIONARY[...]` directly as a default expression would freeze
    those values at import time instead.
    """
    dt_seconds = BLOOD_DICTIONARY["dt_seconds"] if dt_seconds is None else dt_seconds
    K1 = BLOOD_DICTIONARY["K1"] if K1 is None else K1
    k2 = BLOOD_DICTIONARY["k2"] if k2 is None else k2

    dt = dt_seconds / 60.0  # convert to minutes
    t = np.arange(T) * dt

    blood = parker_aif(t)
    tissue = one_tcm_tissue(blood, t, K1=K1, k2=k2)
    background = np.full_like(t, background_level)

    D = np.vstack([blood, tissue, background])
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(
            "One or more dictionary atoms are identically zero and cannot "
            "be normalized; check K1/k2/background_level/dt_seconds."
        )
    D /= norms

    return D  # (3, T)


# ---------------------------
# NNLS Fit for Single Voxel
# ---------------------------
def fit_voxel_to_dictionary(
    tac: NDArray[np.float64], D: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """
    Fit a single voxel's TAC as a non-negative combination of dictionary atoms.

    Parameters
    ----------
    tac : ndarray, shape (T,)
        Voxel time-activity curve.
    D : ndarray, shape (n_atoms, T)
        Dictionary of template TACs.

    Returns
    -------
    w : ndarray, shape (n_atoms,)
        Non-negative mixture weights.
    residual : float
        NNLS residual norm (large values flag voxels poorly explained by
        any combination of dictionary atoms).
    """
    w, residual = nnls(D.T, tac)
    return w, residual


class DictionaryFitResult(NamedTuple):
    weights: NDArray[np.float64]  # (X, Y, Z, n_atoms)
    residuals: NDArray[np.float64]  # (X, Y, Z)


# ---------------------------
# Dictionary Fit for 4D Volume
# ---------------------------
def fit_dictionary_4d(
    c_dyn: NDArray[np.float64],
    num_frames: int,
    smoothing_sigma: float | None = None,
    empty_voxel_tol: float = 1e-8,
    n_jobs: int = 1,
) -> DictionaryFitResult:
    """
    Fit every voxel of a 4D dynamic volume to the fixed TAC dictionary.

    Parameters
    ----------
    c_dyn : ndarray, shape (X, Y, Z, >=num_frames)
        Dynamic image volume.
    num_frames : int
        Number of time frames to use (and to build the dictionary for).
    smoothing_sigma : float, optional
        If given, apply a Gaussian filter with this sigma to `c_dyn` before
        fitting (spatial smoothing only; time axis is left untouched).
    empty_voxel_tol : float
        Voxels whose TAC has L2 norm below this tolerance are treated as
        empty/background and skipped (weights left at zero) rather than
        passed to NNLS. In a typical PET/DCE volume most voxels are outside
        the body, so this is usually a large speedup with no change in the
        result for voxels that actually matter.
    n_jobs : int
        Number of worker processes for the per-voxel NNLS fits. `1` (default)
        runs sequentially. Any other value requires `joblib`; if it isn't
        installed, this falls back to sequential and logs a warning instead
        of raising, since parallelism here is a speed optimization, not a
        correctness requirement.

    Returns
    -------
    DictionaryFitResult
        `weights`: ndarray (X, Y, Z, 3) of NNLS mixture weights.
        `residuals`: ndarray (X, Y, Z) of NNLS residual norms (0 for
        skipped/empty voxels).
    """
    if smoothing_sigma is not None:
        c_dyn = gaussian_filter(c_dyn, sigma=(smoothing_sigma, smoothing_sigma, smoothing_sigma, 0))

    X, Y, Z, _ = c_dyn.shape
    D = build_fixed_dictionary(num_frames)

    tacs = c_dyn[..., :num_frames].reshape(-1, num_frames)
    active_idx = np.flatnonzero(np.linalg.norm(tacs, axis=1) > empty_voxel_tol)
    logger.info(
        "Fitting %d/%d voxels to dictionary (%d skipped as empty/background)",
        len(active_idx),
        tacs.shape[0],
        tacs.shape[0] - len(active_idx),
    )

    W_flat = np.zeros((tacs.shape[0], D.shape[0]), dtype=np.float32)
    residual_flat = np.zeros(tacs.shape[0], dtype=np.float32)

    use_parallel = n_jobs != 1
    if use_parallel and not _HAVE_JOBLIB:
        logger.warning("n_jobs=%s requested but joblib is not installed; running sequentially.", n_jobs)
        use_parallel = False

    if use_parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_voxel_to_dictionary)(tacs[idx], D) for idx in active_idx
        )
        for idx, (w, residual) in zip(active_idx, results):
            W_flat[idx] = w
            residual_flat[idx] = residual
    else:
        for idx in active_idx:
            w, residual = fit_voxel_to_dictionary(tacs[idx], D)
            W_flat[idx] = w
            residual_flat[idx] = residual

    weights = W_flat.reshape(X, Y, Z, D.shape[0])
    residuals = residual_flat.reshape(X, Y, Z)

    return DictionaryFitResult(weights=weights, residuals=residuals)


# ---------------------------
# Extract Largest Vessel
# ---------------------------
def extract_vessel(vessel_input: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Keep only the largest connected component of a binary mask.

    Parameters
    ----------
    vessel_input : ndarray of bool
        Candidate blood mask (typically one hemisphere of the volume).

    Returns
    -------
    ndarray of bool
        Mask containing only the largest connected component. An
        all-False input (or one with no foreground voxels) returns an
        all-False mask rather than raising.
    """
    labels, _ = label(vessel_input)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background label
    if sizes.max() == 0:
        return np.zeros_like(vessel_input, dtype=bool)
    vessel_label = np.argmax(sizes)
    return labels == vessel_label


# ----------------------------
# Morphological erosion
# ----------------------------
def morphological_erosion(
    blood_mask: NDArray[np.bool_], iterations: int = 1
) -> NDArray[np.bool_]:
    """
    Erode a binary blood mask, `iterations` times.

    Parameters
    ----------
    blood_mask : ndarray of bool
        Input mask.
    iterations : int
        Number of erosion iterations (1-2 is typical).

    Returns
    -------
    ndarray of bool
        Eroded mask.

    Notes
    -----
    The structuring element is `(2, 2, 1)`: it erodes in-plane (X, Y) but
    not through-plane (Z), on the assumption that through-plane resolution
    is coarser and thin vessels there are more likely to be real signal
    than partial-volume noise. If your data's voxel spacing is closer to
    isotropic, an isotropic `(2, 2, 2)` (or `scipy`'s default 6-connected)
    structure may be a better fit — worth confirming against your actual
    voxel size rather than assuming this holds for new datasets.
    """
    structure = np.ones((2, 2, 1), dtype=bool)
    return binary_erosion(blood_mask, structure=structure, iterations=iterations)


# ---------------------------
# Anatomical region restriction
# ---------------------------
def _restrict_to_candidate_region(
    blood_mask: NDArray[np.bool_],
    anterior_fraction: float,
    posterior_fraction: float,
) -> NDArray[np.bool_]:
    """
    Zero out voxels outside a plausible blood-vessel region along axis 1.

    Parameters
    ----------
    blood_mask : ndarray of bool, shape (X, Y, Z)
        Candidate mask to restrict, modified in place and returned.
    anterior_fraction : float
        Fraction of the Y extent (from index 0) to zero out — i.e. exclude
        the front `anterior_fraction` of the volume along axis 1.
    posterior_fraction : float
        Fraction of the Y extent (from the far end) to zero out — i.e.
        exclude the back `posterior_fraction` of the volume along axis 1.

    Returns
    -------
    ndarray of bool
        The restricted mask (same array as `blood_mask`, modified in place).

    Notes
    -----
    This assumes axis 1 of the array corresponds to a consistent anatomical
    direction (e.g. anterior-posterior) across all inputs. That's a property
    of image orientation/affine, not of array indexing — if inputs can come
    in with different orientations, this will silently zero out the wrong
    region. Consider reslicing to a canonical orientation before calling
    this, or deriving the cut points from the affine instead of raw indices.
    """
    y_extent = blood_mask.shape[1]
    anterior_cut = int(round(y_extent * anterior_fraction))
    posterior_cut = int(round(y_extent * posterior_fraction))

    blood_mask[:, :anterior_cut, :] = False
    if posterior_cut > 0:
        blood_mask[:, -posterior_cut:, :] = False
    return blood_mask


# ---------------------------
# Generate Blood Mask (pure computation, no I/O)
# ---------------------------
def compute_blood_mask(
    img: NDArray[np.float64],
    num_frames: int,
    smoothing_sigma: float | None = 0,
    apply_erosion: bool = False,
    erosion_iterations: int = 1,
    anterior_fraction: float = 0.5,
    posterior_fraction: float = 0.1,
    n_jobs: int = 1,
) -> NDArray[np.bool_]:
    """
    Identify blood voxels in a dynamic 4D volume.

    Parameters
    ----------
    img : ndarray, shape (X, Y, Z, T)
        Dynamic image volume.
    num_frames : int
        Number of time frames to use for the dictionary fit.
    smoothing_sigma : float, optional
        Spatial Gaussian smoothing sigma applied before fitting. `None`
        disables smoothing.
    apply_erosion : bool
        If True, erode the final mask by `erosion_iterations` voxels to
        strip thin, likely-partial-volume edges. Off by default, since
        erosion trades away small/peripheral vessels for a cleaner core.
    erosion_iterations : int
        Number of erosion iterations, used only if `apply_erosion=True`.
    anterior_fraction, posterior_fraction : float
        Passed to `_restrict_to_candidate_region` — see that function's
        docstring, including the orientation caveat.
    n_jobs : int
        Passed to `fit_dictionary_4d` for parallel per-voxel fitting.

    Returns
    -------
    ndarray of bool, shape (X, Y, Z)
        Final blood mask.
    """
    fit = fit_dictionary_4d(img, num_frames, smoothing_sigma=smoothing_sigma, n_jobs=n_jobs)
    W_map = fit.weights

    threshold = BLOOD_DICTIONARY["blood_threshold"]
    is_blood = W_map[..., 0] >= threshold
    is_tissue = W_map[..., 1] >= threshold
    is_background = W_map[..., 2] >= threshold
    blood_mask = is_blood & ~is_tissue & ~is_background

    blood_mask = _restrict_to_candidate_region(
        blood_mask, anterior_fraction=anterior_fraction, posterior_fraction=posterior_fraction
    )

    x_mid = blood_mask.shape[0] // 2
    blood_mask[:x_mid, :, :] = extract_vessel(blood_mask[:x_mid, :, :])
    blood_mask[x_mid:, :, :] = extract_vessel(blood_mask[x_mid:, :, :])

    if apply_erosion:
        blood_mask = morphological_erosion(blood_mask, iterations=erosion_iterations)

    return blood_mask


def save_mask(mask: NDArray[np.bool_], affine: NDArray[np.float64], path: str | Path) -> Path:
    """
    Save a boolean mask as `<path>/blood.nii`.

    Parameters
    ----------
    mask : ndarray of bool
        Mask to save.
    affine : ndarray, shape (4, 4)
        NIfTI affine transform.
    path : str or Path
        Output directory (must already exist).

    Returns
    -------
    Path
        Full path of the written file.
    """
    out_dir = Path(path)
    out_path = out_dir / "blood.nii"
    array_img = nib.Nifti1Image(mask.astype(np.int32), affine)
    nib.save(array_img, str(out_path))
    return out_path


def make_blood_mask(
    path: str,
    img: NDArray[np.float64],
    affine: NDArray[np.float64],
    num_frames: int,
    smoothing_sigma: float | None = None,
    apply_erosion: bool = False,
    erosion_iterations: int = 1,
) -> NDArray[np.bool_]:
    """
    Identify blood voxels in a dynamic 4D volume and save the mask as NIfTI.

    Thin convenience wrapper kept for backward compatibility with existing
    call sites; combines `compute_blood_mask` and `save_mask`. Prefer calling
    those two directly in new code (e.g. in tests) so the computation can be
    exercised without touching disk.

    Parameters are as in `compute_blood_mask`, plus:

    path : str
        Output directory; the mask is written to `<path>/blood.nii`.

    Returns
    -------
    ndarray of bool, shape (X, Y, Z)
        Final blood mask (also written to disk).
    """
    blood_mask = compute_blood_mask(
        img,
        num_frames,
        smoothing_sigma=smoothing_sigma,
        apply_erosion=apply_erosion,
        erosion_iterations=erosion_iterations,
    )
    save_mask(blood_mask, affine, path)
    return blood_mask