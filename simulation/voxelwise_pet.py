"""
Voxel-wise irreversible 2-tissue-compartment (2TCM, k4=0) NLLS fitting for
dynamic PET -- the PET counterpart of `dce.analysis.calculate_dce_voxelwise`
(which only implements the 1TCM case). Not present anywhere in the
original repo; needed here as the "voxelwise" baseline for the PET arm of
the validation.

Uses the same closed-form solution as `core.model.Ks_net.convolve_2cm_for_minimize`
/ `simulation.forward_models.simulate_2tcm_volume`, fit per voxel with
scipy's derivative-free Powell method (matching the style of the existing
1TCM voxelwise fitter).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def one_tcm_solution(t, K1, k2, aif_interp):
    cp = aif_interp(t)
    return K1 * exp_conv_trap(t, cp, k2)


def fit_1tcm_voxel(tac, t, aif_interp, p0=(0.2, 0.3), bounds=((1e-6, 3.0), (1e-6, 3.0))):
    """PET 1TCM voxelwise fit -- same math as dce.analysis.fit_1tcm_ode_voxel,
    reproduced here (rather than imported) so this module has no
    dependency on the dce package, and with PET-appropriate default
    bounds (PET K1/k2 commonly exceed the DCE fitter's default 1.0 cap,
    e.g. the literature K1/k2 ranges used in kinetics_literature.py)."""
    if tac.max() < 1e-6:
        return np.nan, np.nan

    def objective(params):
        K1, k2 = params
        model = one_tcm_solution(t, K1, k2, aif_interp)
        return np.sum((model - tac) ** 2)

    try:
        result = minimize(objective, x0=p0, bounds=bounds, method="Powell")
        return tuple(result.x)
    except Exception:
        return np.nan, np.nan


def calculate_pet_voxelwise_1tcm(img, t, aif, mask=None, verbose=True):
    """
    1-tissue-compartment (K1, k2 only -- no k3) voxelwise NLLS fit for
    PET, e.g. for a short acquisition where a 2TCM's k3 isn't expected
    to be identifiable (see the 20-min-vs-60-min discussion this
    session) or as a fast/simple baseline alongside the full 2TCM fit
    above.

    img:  (T, X, Y, Z)
    t:    (T,) frame timestamps
    aif:  (T,) arterial input sampled at t
    mask: optional (X, Y, Z) bool array restricting the fit

    Returns dict of (X, Y, Z) maps: K1, k2.
    """
    T, X, Y, Z = img.shape
    flat = img.reshape(T, -1).T

    flat_mask = mask.reshape(-1) if mask is not None else flat.max(axis=1) > 1e-6
    valid_idx = np.where(flat_mask)[0]
    aif_interp = interp1d(t, aif, kind="linear", fill_value="extrapolate")

    K1_flat = np.full(X * Y * Z, np.nan)
    k2_flat = np.full(X * Y * Z, np.nan)

    for n, idx in enumerate(valid_idx):
        if verbose and n % 500 == 0:
            print(f"[voxelwise-PET-1TCM] {n}/{len(valid_idx)}")
        tac = flat[idx]
        K1, k2 = fit_1tcm_voxel(tac, t, aif_interp)
        K1_flat[idx], k2_flat[idx] = K1, k2

    return {"K1": K1_flat.reshape(X, Y, Z), "k2": k2_flat.reshape(X, Y, Z)}


def exp_conv_trap(t, blood, theta):
    dt = np.diff(t, prepend=t[0])
    H = np.zeros_like(blood)
    for i in range(1, len(t)):
        alpha = np.exp(-theta * dt[i])
        H[i] = alpha * H[i - 1] + 0.5 * dt[i] * (blood[i] + alpha * blood[i - 1])
    return H


def two_tcm_solution(t, K1, k2, k3, aif_interp):
    cp = aif_interp(t)
    delta = max(k2 + k3, 1e-8)
    theta1, theta2 = delta, 0.0
    phi1 = K1 * (theta1 - k3) / delta
    phi2 = K1 * (theta2 - k3) / (-delta)
    H1 = exp_conv_trap(t, cp, theta1)
    H2 = exp_conv_trap(t, cp, theta2)
    return phi1 * H1 + phi2 * H2


def fit_2tcm_voxel(tac, t, aif_interp, p0=(0.2, 0.3, 0.05),
                    bounds=((1e-6, 2.0), (1e-6, 2.0), (1e-6, 2.0))):
    if tac.max() < 1e-6:
        return np.nan, np.nan, np.nan

    def objective(params):
        K1, k2, k3 = params
        model = two_tcm_solution(t, K1, k2, k3, aif_interp)
        return np.sum((model - tac) ** 2)

    try:
        result = minimize(objective, x0=p0, bounds=bounds, method="Powell")
        return tuple(result.x)
    except Exception:
        return np.nan, np.nan, np.nan


def calculate_pet_voxelwise(img, t, aif, mask=None, verbose=True):
    """
    img:  (T, X, Y, Z)
    t:    (T,) frame timestamps (NOT durations -- see note in validate.py
          about pet/analysis.py passing durations directly)
    aif:  (T,) arterial input sampled at t
    mask: optional (X, Y, Z) bool array restricting the fit

    Returns dict of (X, Y, Z) maps: K1, k2, k3.
    """
    T, X, Y, Z = img.shape
    flat = img.reshape(T, -1).T  # (N, T)

    flat_mask = mask.reshape(-1) if mask is not None else flat.max(axis=1) > 1e-6
    valid_idx = np.where(flat_mask)[0]
    aif_interp = interp1d(t, aif, kind="linear", fill_value="extrapolate")

    K1_flat = np.full(X * Y * Z, np.nan)
    k2_flat = np.full(X * Y * Z, np.nan)
    k3_flat = np.full(X * Y * Z, np.nan)

    for n, idx in enumerate(valid_idx):
        if verbose and n % 500 == 0:
            print(f"[voxelwise-PET] {n}/{len(valid_idx)}")
        tac = flat[idx]
        K1, k2, k3 = fit_2tcm_voxel(tac, t, aif_interp)
        K1_flat[idx], k2_flat[idx], k3_flat[idx] = K1, k2, k3

    return {
        "K1": K1_flat.reshape(X, Y, Z),
        "k2": k2_flat.reshape(X, Y, Z),
        "k3": k3_flat.reshape(X, Y, Z),
    }
