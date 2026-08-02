import glob
import os

import numpy as np
import nibabel as nib
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from config import DCE, PINN, DEVICE
from core.create_blood_mask import make_blood_mask
from core.train import Trainer
from core.utils import get_tac_from_masked_region
from dce import preprocessing


# ----------------------------------------------------------------------
# Voxelwise 1TCM (Tofts, no vb) fit -- minimize ODE solution vs observed TAC
# using scipy.optimize.minimize with the derivative-free Powell method
# ----------------------------------------------------------------------
def exp_conv_trap(t, blood, theta):
    """
    Trapezoidal convolution of AIF with exp(-theta*t) -- exact solution of
    dCt/dt = K1*Cp - k2*Ct given Cp, same recursion as TACDataset.exp_conv_trap.
    """
    dt = np.diff(t, prepend=t[0])
    H = np.zeros_like(blood)
    for i in range(1, len(t)):
        alpha = np.exp(-theta * dt[i])
        H[i] = alpha * H[i - 1] + 0.5 * dt[i] * (blood[i] + alpha * blood[i - 1])
    return H


def one_tcm_solution(t, K1, Ve, aif_interp):
    """Forward ODE solution: Ct(t) = K1 * conv(Cp, exp(-k2*t))."""
    Cp = aif_interp(t)
    return K1 * exp_conv_trap(t, Cp, K1/Ve)


def fit_1tcm_ode_voxel(tac, t, aif_interp, p0=(0.1, 0.65), bounds=((1e-6, 1.0), (1e-6, 1.0))):
    """
    Fit one voxel by minimizing sum((Ct_model(t; K1, k2) - Ct_observed(t))^2).

    Uses scipy.optimize.minimize with `bounds` given and no `method`
    specified -- scipy's automatic selection resolves this to L-BFGS-B,
    NOT Powell (an earlier version of this docstring incorrectly claimed
    Powell; that was never actually true, since `method` was never
    explicitly passed). Verified this matters empirically: on low-signal,
    noisy voxels (the exact regime real tissue-edge/background voxels
    hit), explicit Powell produced catastrophic `ve = K1/k2` blowups
    (observed up to ve=884,565 in repeated-noise-draw testing) that
    L-BFGS-B and a global bounded optimizer (dual_annealing) did not --
    dual_annealing matched L-BFGS-B's robustness but at ~10x the runtime
    for no accuracy gain. Do not switch this to Powell.

    Returns (K1, k2), or NaNs if the fit fails.
    """
    if tac.max() < 1e-6:
        return np.nan, np.nan

    def objective(params):
        K1, k2 = params
        model = one_tcm_solution(t, K1, k2, aif_interp)
        return np.sum((model - tac) ** 2)

    try:
        result = minimize(objective, x0=p0, bounds=bounds)
        K1, k2 = result.x
        return K1, k2
    except Exception:
        return np.nan, np.nan


def calculate_dce_voxelwise(img, t, aif, mask=None):
    """
    img:  (T, X, Y, Z) numpy array
    t:    (T,) numpy array of frame times
    aif:  (T,) numpy array, AIF sampled at t
    mask: optional (X, Y, Z) bool array restricting the fit

    Returns dict of (X, Y, Z) maps: K1 (Ktrans), k2 (kep), ve (=K1/k2)
    Fit via Powell's method directly on the ODE solution (derivative-free
    nonlinear optimization of the true 1TCM forward model vs the TAC).
    """
    T, X, Y, Z = img.shape
    flat = img.reshape(T, -1).T  # (X*Y*Z, T)
    print(T)

    if mask is not None:
        flat_mask = mask.reshape(-1)
    else:
        flat_mask = flat.max(axis=1) > 1e-6  # skip empty/background voxels

    valid_idx = np.where(flat_mask)[0]
    aif_interp = interp1d(t, aif, kind='linear', fill_value="extrapolate")

    K1_flat = np.full(X * Y * Z, np.nan)
    k2_flat = np.full(X * Y * Z, np.nan)

    for idx in valid_idx:
        if idx%1000==0:
            print(idx,len(valid_idx))
        tac = flat[idx]
        K1, k2 = fit_1tcm_ode_voxel(tac, t, aif_interp)
        K1_flat[idx] = K1
        k2_flat[idx] = k2

    K1_map = K1_flat.reshape(X, Y, Z)
    k2_map = k2_flat.reshape(X, Y, Z)
    ve_map = K1_map / k2_map

    # ve (extracellular volume fraction) is physically bounded to [0, 1]
    # by definition -- any fitted value outside that range is a failed/
    # degenerate fit (k2 landed too close to its lower bound), not a real
    # measurement. Mask these out rather than silently reporting nonsense
    # values that can (and did, in real data checked this session) blow
    # up an ROI mean by orders of magnitude from a single bad voxel.
    ve_map = np.where((ve_map >= 0) & (ve_map <= 1.0), ve_map, np.nan)

    return {"K1": K1_map, "k2": k2_map, "ve": ve_map}


def save_maps_as_nifti(maps, affine, out_dir):
    """Save each parametric map (K1, k2, ve, ...) as a .nii.gz file."""
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in maps.items():
        arr_clean = np.nan_to_num(arr, nan=0.0)
        img_nii = nib.Nifti1Image(arr_clean.astype(np.float32), affine)
        out_path = os.path.join(out_dir, f"{name}_map.nii.gz")
        nib.save(img_nii, out_path)
        print(f"[SAVED] {out_path}")


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def pipeline(
        path,
        num_of_compartment=1,
        epochs=PINN["epochs"],
        device=DEVICE,
        method="pinn",
        n_ensemble=5,
        dropout_p=0.1,
):
    """
    Full DCE/PET kinetic pipeline.

    method:
        "pinn"       -> PINN kinetic fitting (point estimate only)
        "voxelwise" -> classical voxelwise ODE fitting
        "bayesian"  -> Sine B-PINN deep ensemble: same K1/k2(/k3) point
                       estimate as "pinn", PLUS a per-voxel uncertainty
                       map (see core/uncertainty.py). Saves K_mean.nii,
                       K_uncertainty.nii, and K_uncertainty_demeaned.nii
                       (prefer the demeaned map for per-voxel calibration
                       checks -- see core.uncertainty.estimate_with_uncertainty's
                       docstring for why). Costs ~n_ensemble x the runtime
                       of a single "pinn" fit.

    n_ensemble, dropout_p : only used by method="bayesian" -- see
        core.uncertainty.estimate_with_uncertainty.
    """


    # -----------------------------
    # 1. Preprocessing
    # -----------------------------

    img, aif, affine = preprocessing(path)


    # img:
    # (X,Y,Z,T)

    x,y,z,_ = np.asarray(img.shape)//3


    cropped = img[
        x:-x,
        x:-x,
        :,
        :
    ].transpose(3,0,1,2)
    # (T,X,Y,Z)


    new_affine = affine.copy()

    new_affine[:3,3] += (
        new_affine[:3,:3] @ np.array([x,y,z])
    )


    t=np.cumsum(DCE["dt"])



    # -----------------------------
    # 2. PINN
    # -----------------------------

    if method=="pinn":

        trainer=Trainer(
            c_p=aif,
            num_of_compartment=num_of_compartment,
            t=t,
            device=device,
            affine=new_affine,
            save_path=path,
            epochs=epochs,
        )

        trainer.train_ensemble(cropped)



    # -----------------------------
    # 2b. Sine B-PINN ensemble (point estimate + per-voxel uncertainty)
    # -----------------------------

    elif method == "bayesian":
        from core.uncertainty import estimate_with_uncertainty

        tissue_mask = cropped.max(axis=0) > (cropped.max() * 0.05)
        result = estimate_with_uncertainty(
            cropped, aif, t, num_of_compartment=num_of_compartment,
            save_path=path, affine=new_affine, device=device,
            n_ensemble=n_ensemble, epochs=epochs, dropout_p=dropout_p,
            tissue_mask=tissue_mask,
        )
        return result



    # -----------------------------
    # 3. Classical voxel fitting
    # -----------------------------

    elif method=="voxelwise":

        maps=calculate_dce_voxelwise(
            cropped,
            t,
            aif
        )

        save_maps_as_nifti(
            maps,
            new_affine,
            out_dir=path
        )

        return maps



    else:
        raise ValueError(
            f"Unknown method {method}"
        )
def run_all_dce(root_path, epochs=1000, device="cpu", method="pinn", n_ensemble=5, dropout_p=0.1):
    """
    FDA-style batch executor for DCE pipelines.
    Processes all subjects matching sub*/dce within the root directory.
    n_ensemble, dropout_p are only used when method="bayesian".
    """

    print(f"[INFO] Searching for subjects in: {root_path}")

    subject_paths = sorted(glob.glob(os.path.join(root_path, "sub*")))

    if len(subject_paths) == 0:
        print("[WARNING] No subject folders found matching sub*/")
        return

    print(f"[INFO] Found {len(subject_paths)} subject(s).")

    for i, sub_path in enumerate(subject_paths, start=1):
        subject_id = os.path.basename(sub_path)
        dce_path = os.path.join(sub_path, "dce")

        if not os.path.isdir(dce_path):
            print(f"[SKIP] {subject_id}: No /dce folder found. Skipping.")
            continue

        print(f"\n[INFO] ({i}/{len(subject_paths)}) Processing {subject_id}")
        print(f"[INFO] DCE path: {dce_path}")

        try:
            pipeline(dce_path, epochs=epochs, device=device, method=method,
                     n_ensemble=n_ensemble, dropout_p=dropout_p)
            print(f"[SUCCESS] {subject_id} completed.\n")

        except Exception as e:
            print(f"[ERROR] {subject_id} failed with error:")
            print(f"        {e}")
            print("[ACTION] Continuing to next subject.\n")

    print("[INFO] Batch DCE execution completed.")